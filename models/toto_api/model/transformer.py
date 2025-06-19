# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import warnings
from typing import Literal, Optional, Union, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Int
from rotary_embedding_torch import RotaryEmbedding

from models.toto_api.model.attention import (
    AttentionAxis,
    MultiHeadAttention,
    SpaceWiseMultiheadAttention,
    TimeWiseMultiheadAttention,
)
from models.toto_api.model.feed_forward import SwiGLU
from models.toto_api.model.rope import TimeAwareRotaryEmbedding
from models.toto_api.model.util import KVCache, RMSNorm, make_batched_block_mask

try:
    from xformers.ops.swiglu_op import SwiGLU as SwiGLU_fused

    XFORMERS_SWIGLU_AVAILABLE = True
except ImportError:
    warnings.warn(
        "xFormers fused SwiGLU kernel not found. " "Using native PyTorch implementation for feed-forward layers.",
        ImportWarning,
    )
    XFORMERS_SWIGLU_AVAILABLE = False


class TransformerLayer(torch.nn.Module):
    """
    A transformer block that applies multihead attention followed by a feedforward network.

    The transformer can be configured to apply time-wise attention (i.e. attention over the time axis)
    or space-wise attention (i.e. attention over the variate axis).

    The transformer block uses pre-norm, which is a variant of the transformer architecture where
    LayerNorm is applied before each sublayer, rather than after. This is the approach taken in
    LLaMA and other recent transformer-based models.

    The transformer block also uses SwiGLU, which is a variant of the Gated Linear Unit (GLU) activation
    function. SwiGLU is a variant of the GLU activation that uses the Swish activation function. This
    activation function has been used extensively in recent transformer-based models and has been shown
    to improve performance.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        rotary_emb: RotaryEmbedding = None,
        attention_axis: AttentionAxis = AttentionAxis.TIME,
        RMS_norm: bool = True,
        use_memory_efficient_attention: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout = dropout
        self.attention_axis = attention_axis

        if RMS_norm:
            self.norm1 = RMSNorm(embed_dim)
            self.norm2 = RMSNorm(embed_dim)

        else:
            self.norm1 = torch.nn.LayerNorm(embed_dim)
            self.norm2 = torch.nn.LayerNorm(embed_dim)

        self.attention: MultiHeadAttention

        if attention_axis == AttentionAxis.TIME:
            self.attention = TimeWiseMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                rotary_emb=rotary_emb,
                use_memory_efficient_attention=use_memory_efficient_attention,
            )
        elif attention_axis == AttentionAxis.SPACE:
            self.attention = SpaceWiseMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                rotary_emb=None,
                use_memory_efficient_attention=use_memory_efficient_attention,
            )
        else:
            raise ValueError("Invalid attention axis")

        if XFORMERS_SWIGLU_AVAILABLE:
            self.mlp = torch.nn.Sequential(
                SwiGLU_fused(in_features=embed_dim, hidden_features=mlp_hidden_dim),
                torch.nn.Dropout(dropout),
            )
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, 2 * mlp_hidden_dim),
                SwiGLU(),
                torch.nn.Linear(mlp_hidden_dim, embed_dim),
                torch.nn.Dropout(dropout),
            )

    def forward(
        self,
        layer_idx: int,
        inputs: Float[torch.Tensor, "batch variate seq_len embed_dim"],
        attention_mask: Optional[
            Union[
                Bool[torch.Tensor, "batch seq_len variate variate"],
                Bool[torch.Tensor, "batch #variate seq_len seq_len"],
            ]
        ] = None,
        kv_cache: KVCache = None,
    ) -> Float[torch.Tensor, "batch variate seq_len embed_dim"]:
        pre_norm_1 = self.norm1(inputs)
        hidden_state = inputs + self.attention(layer_idx, pre_norm_1, attention_mask, kv_cache).contiguous()

        pre_norm_2 = self.norm2(hidden_state)
        return hidden_state + self.mlp(pre_norm_2)


class Transformer(torch.nn.Module):
    """
    A stack of transformer layers. The transformer alternates between time-wise and space-wise attention
    to learn both temporal and cross-variate dependencies in the data.

    Based on the intuition that time-wise attention is more important overall than space-wise attention
    (because an individual variate is more likely to be correlated with itself across time than with other variates),
    the transformer can be configured to apply space-wise attention less frequently than time-wise attention.
    This is controlled by the `spacewise_every_n_layers` parameter, which specifies how many time-wise transformer
    layers to apply between every space-wise transformer layer.

    Parameters
    ----------
    num_layers
        Number of transformer layers to use.
    num_heads
        Number of attention heads to use in each self-attention layer.
    mlp_hidden_dim
        Dimension of the hidden layer in the feedforward network.
    dropout
        Dropout rate to use in the model.
    spacewise_every_n_layers
        How many time-wise transformer layers to apply between each space-wise transformer layer.
    spacewise_first
        Whether to apply space-wise attention before time-wise attention.
    use_memory_efficient_attention
        Whether to use memory-efficient attention. If True, the model will use the memory-efficient from xFormers.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        spacewise_every_n_layers: int,
        spacewise_first: bool,
        use_memory_efficient_attention: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.rotary_emb = TimeAwareRotaryEmbedding(
            embed_dim // num_heads,
            use_xpos=True,
            cache_if_possible=True,
            seq_before_head_dim=use_memory_efficient_attention,
        )
        attention_axes = self._get_layer_types(num_layers, spacewise_every_n_layers, spacewise_first)

        self.use_memory_efficient_attention = use_memory_efficient_attention

        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                    rotary_emb=self.rotary_emb,
                    attention_axis=attention_axes[i],
                    use_memory_efficient_attention=self.use_memory_efficient_attention,
                )
                for i in range(num_layers)
            ]
        )

    def _get_mask(
        self,
        num_heads: int,
        dtype: torch.dtype,
        id_mask: Optional[torch.Tensor] = None,
    ) -> Union[
        Bool[torch.Tensor, "batch num_heads seq_len seq_len"],
        Float[torch.Tensor, "batch num_heads seq_len seq_len"],
        Bool[torch.Tensor, "batch num_heads variate variate"],
        Float[torch.Tensor, "batch num_heads variate variate"],
    ]:
        """
        Unified method to create and process space-wise masks.

        Args:
            mask_type: Type of mask to create ('spacewise').
            seq_len: Total sequence length.
            num_heads: Number of attention heads.
            device: Device where the mask should be created.
            dtype: Desired dtype for the bias tensor.
            id_mask: Mask for variates (used for spacewise masks).

        Returns:
            Processed attention mask tensor with the correct shape for the given mask type.
        """

        if id_mask is None:
            raise ValueError("id_mask must be provided for spacewise masks.")

        # Create spacewise mask
        mask = make_batched_block_mask(id_mask.transpose(-1, -2))

        if self.use_memory_efficient_attention:
            mask = self._pad_to_multiple(mask)
            mask = mask.float().masked_fill(~mask, float("-inf")).masked_fill(mask, 0.0).to(dtype)

        # Rearrange for space-wise attention
        mask = rearrange(mask, "batch seq_len variate1 variate2 -> (batch seq_len) 1 variate1 variate2")
        # Stack along num_heads dimension
        return mask.expand(-1, num_heads, -1, -1).contiguous()

    def _pad_to_multiple(
        self,
        tensor: torch.Tensor,
        multiple: int = 8,
        causal: bool = False,  # New flag to indicate causal mask extension
    ) -> torch.Tensor:
        """
        Pads the last two dimensions of a tensor to be divisible by `multiple`.
        For causal masks, the padded area is filled with the continued lower-triangular pattern,
        rather than with zeros.
        """
        pad_amount = (multiple - tensor.shape[-1] % multiple) % multiple
        if pad_amount > 0:
            new_size = tensor.shape[-1] + pad_amount
            if causal:
                # Create a full causal mask for the new size.
                full_mask = torch.tril(torch.ones((new_size, new_size), dtype=tensor.dtype, device=tensor.device))
                # Preserve any modifications from the original mask (e.g., condition tokens in top-left)
                full_mask[: tensor.shape[-1], : tensor.shape[-1]] = tensor
                tensor = full_mask
            else:
                tensor = F.pad(tensor, (0, pad_amount, 0, pad_amount))
        return tensor

    def _get_layer_types(
        self,
        num_layers: int,
        spacewise_every_n_layers: int,
        spacewise_first: bool,
    ) -> list[AttentionAxis]:
        if spacewise_every_n_layers == -1:
            return [AttentionAxis.TIME] * num_layers
        assert num_layers % spacewise_every_n_layers == 0

        block = [AttentionAxis.TIME] * (spacewise_every_n_layers - 1)

        if spacewise_first:
            block = [AttentionAxis.SPACE] + block
        else:
            block = block + [AttentionAxis.SPACE]

        layer_types = block * (num_layers // spacewise_every_n_layers)

        return layer_types

    def forward(
        self,
        inputs: Float[torch.Tensor, "batch variate seq_len embed_dim"],
        id_mask: Float[torch.Tensor, "batch #variate seq_len"],
        kv_cache: KVCache = None,
    ) -> Float[torch.Tensor, "batch variate seq_len embed_dim"]:

        batch, _, seq_len, _ = inputs.shape
        # Get the sequence length by looking up a timewise layer in the kv cache.
        # Regardless of whether spacewise is first in the stack, the layer
        # at index 1 is always a timewise layer.
        seq_len = (kv_cache.seq_len(1) if kv_cache else 0) + seq_len

        num_heads: int = cast(int, self.layers[0].num_heads)

        timewise_attention_mask = None

        # We create a space-wise ID mask by creating a block triangular mask from the ID mask
        # in the space-wise direction. This ensures that the model can only attend to
        # variates in the same group.
        spacewise_attention_mask = self._get_mask(
            num_heads=num_heads,
            dtype=inputs.dtype,
            id_mask=id_mask,
        )

        for layer_idx, layer in enumerate(self.layers):
            inputs = layer(
                layer_idx,
                inputs,
                (timewise_attention_mask if layer.attention_axis == AttentionAxis.TIME else spacewise_attention_mask),
                kv_cache,
            )
        return inputs

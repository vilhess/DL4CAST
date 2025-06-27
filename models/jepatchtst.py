import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
import math
import lightning as L 
from typing import Optional
from huggingface_hub import PyTorchModelHubMixin

from models.metric import StreamMAELoss, StreamMSELoss

class Patcher(nn.Module):
    def __init__(self, window_size, patch_len):
        super().__init__()
        assert window_size % patch_len == 0, "window size must be divisible by patch length"
        self.window_size = window_size
        self.patch_len = patch_len
        self.patch_num = window_size// patch_len
        self.shape = {"window_size":self.window_size,
                              "patch_len":self.patch_len,
                              "patch_num":self.patch_num}

    def forward(self, window):

        # Input: 

        # x: bs x nvars x window_size

        # Output:

        # out: bs x nvars x patch_num x patch_len 
        patch_window = rearrange(window, 'b c (pn pl) -> b c pn pl', pl=self.patch_len)
        return patch_window
    
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

def PositionalEncoding(q_len, d_model):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return nn.Parameter(pe, requires_grad=False)

class RotaryPositionalEmbeddings(nn.Module):
    """
    Code copied from https://docs.pytorch.org/torchtune/0.4/_modules/torchtune/modules/position_embeddings.html

    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = x.size(1)

        if input_pos is not None:
            bs = input_pos.size(0)
            bs_n_vars = x.size(0)
            n_vars = bs_n_vars // bs
            if n_vars > 1:
                input_pos = input_pos.repeat_interleave(n_vars, dim=0)

        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3)
        return x_out.type_as(x)



class _ScaledDotProduct(nn.Module):
    def __init__(self, d_model, n_heads, attn_dp=0.):
        super().__init__()

        self.attn_dp = nn.Dropout(attn_dp)
        head_dim = d_model//n_heads
        self.scale = head_dim**(-0.5)

    def forward(self, q, k, v, prev=None):
        
        # Input: 

        # q: bs x nheads x num_patches x d_k
        # k: bs x nheads x d_k x num_patches
        # v: bs x nheads x num_patches x d_v
        # prev: bs x nheads x num_patches x num_patches

        # Output:

        # out: bs x nheads x num_patches x d_v
        # attn_weights: bs x nheads x num_patches x num_patches
        # attn_scores: bs x nheads x num_patches x num_patches

        attn_scores = torch.matmul(q, k)*self.scale

        if prev is not None: attn_scores+=prev

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dp(attn_weights)

        out = torch.matmul(attn_weights, v)
        
        return out, attn_scores
    

class _MultiHeadAttention(nn.Module):
    def __init__(self, patch_num, d_model, n_heads, d_k=None, d_v=None, attn_dp=0., proj_dp=0., qkv_bias=True):
        super().__init__()
        d_k = d_model//n_heads if d_k is None else d_k
        d_v = d_model//n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, n_heads*d_k, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, n_heads*d_k, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, n_heads*d_v, bias=qkv_bias)

        self.sdp = _ScaledDotProduct(d_model=d_model, n_heads=n_heads, attn_dp=attn_dp)

        self.to_out = nn.Sequential(nn.Linear(n_heads*d_v, d_model), nn.Dropout(proj_dp))

        self.rotatory_embedding = RotaryPositionalEmbeddings(dim=d_k, max_seq_len=patch_num)

    def forward(self, Q, K=None, V=None, prev=None):

        # Input: 

        # Q: bs x num_patches x d_model
        # K: bs x num_patches x d_model
        # V: bs x num_patches x d_model
        # prev: bs x num_patches x num_patches

        # Output:

        # out: bs x num_patches x d_model
        # attn_scores: bs x num_patches x num_patches

        bs = Q.size(0)
        if K is None: K = Q.clone()
        if V is None: V = Q.clone()

        q = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k)
        k = self.W_K(K).view(bs, -1, self.n_heads, self.d_k)
        v = self.W_V(V).view(bs, -1, self.n_heads, self.d_v)

        q = self.rotatory_embedding(q)  
        k = self.rotatory_embedding(k)

        q = q.transpose(1, 2)
        k = k.permute(0, 2, 3, 1) 
        v = v.transpose(1, 2)

        out, attn_scores = self.sdp(q, k, v, prev=prev)

        out = out.transpose(1, 2).contiguous().view(bs, -1, self.n_heads*self.d_v)
        out = self.to_out(out)

        return out, attn_scores
    
class TSTEncoderLayer(nn.Module):
    def __init__(self, patch_num, d_model, n_heads, d_k=None, d_v=None, d_ff=256, attn_dp=0., dp=0.):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.self_attn = _MultiHeadAttention(patch_num=patch_num, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, attn_dp=attn_dp, proj_dp=dp)
        self.attn_dp = nn.Dropout(attn_dp)
        self.norm_attn = nn.RMSNorm([d_model])
        
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(dp),
                                nn.Linear(d_ff, d_model))
        
        self.ffn_dp = nn.Dropout(dp)
        self.norm_ffn = nn.RMSNorm([d_model])

    def forward(self, src, prev):

        # Input: 

        # src: bs x num_patches x d_model
        # prev: bs x n_heads x num_patches x num_patches

        # Output:

        # out: bs x num_patches x d_model
        # attn_scores: bs x nheads x num_patches x num_patches

        src, scores = self.self_attn(Q=src, prev=prev)
        src = self.attn_dp(src)
        src = self.norm_attn(src)

        src2 = self.ff(src)

        src = src + self.ffn_dp(src2)
        src = self.norm_ffn(src)

        return src, scores
    

class TSTEncoder(nn.Module):
    def __init__(self, patch_num, d_model, n_heads, d_k=None, d_v=None, d_ff=256, attn_dp=0., dp=0., n_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([TSTEncoderLayer(patch_num=patch_num, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, 
                                                     d_ff=d_ff, attn_dp=attn_dp, dp=dp) for _ in range(n_layers)])
        
    def forward(self, x):

        # Input: 

        # x: bs x num_patches x d_model

        # Output:

        # out: bs x num_patches x d_model
        out=x
        prev=None
        for layer in self.layers:
            out, prev = layer(out, prev=prev)
        return out
    

class TSTiEncoder(nn.Module):
    def __init__(self, patch_num, patch_len, d_model, n_heads, n_layers=3, d_ff=256, attn_dp=0., dp=0.):
        super().__init__()
        self.patch_num, self.patch_len = patch_num, patch_len

        self.W_P = nn.Linear(patch_len, d_model)
        self.dp=nn.Dropout(dp)
        
        self.encoder = TSTEncoder(patch_num=patch_num, d_model=d_model, n_heads=n_heads, d_ff=d_ff, attn_dp=attn_dp, dp=dp, n_layers=n_layers)

    def forward(self, x):

        # Input: 

        # x: bs x nvars x num_patches  x patch_len

        # Output:

        # out: bs x nvars x d_model x num_patches

        n_vars = x.shape[1]
        x = self.W_P(x) # bs x nvars x num_patches x d_model

        x = torch.reshape(x, (x.shape[0]*x.shape [1], x.shape[2], x.shape[3])) # bs*nvars x num_patches x d_model    (channel indep)
        x = self.dp(x)
        x = self.encoder(x) # bs*nvars x num_patches x d_model
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1])) # bs x nvars x num_patches x d_model

        return x  # bs x nvars x num_patches x d_model
    

class JEPAtchTSTEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        window_size = config["ws"]
        patch_len = config["patch_len"]
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        d_ff = config["d_ff"]
        attn_dp=0.1
        dp=0.1

        self.patcher = Patcher(window_size=window_size, patch_len=patch_len)
        shape = self.patcher.shape
        patch_num = shape["patch_num"]

        self.encoder = TSTiEncoder(patch_num=patch_num, patch_len=patch_len, d_model=d_model, 
                                   n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, attn_dp=attn_dp,
                                   dp=dp)
        self.tp = Transpose(1, 2)
    

    def forward(self, x):
        # Input: 

        # x: bs x window_size x nvars
        
        patched = self._get_patch(x) # bs x nvars x patch_len x patch_num
        
        h = self.encoder(patched) # bs x nvars x patch_num x d_model

        return patched, h
    
    def _get_patch(self, x):
        x = self.tp(x) # bs x nvars x window_size
        patched = self.patcher(x) # bs x nvars x patch_num x patch_len
        return patched
    

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    
class PredictorHead(nn.Module):
    def __init__(self, n_vars, patch_num,  d_model, target_len, head_dp=0.):
        super().__init__() 

        self.n_vars = n_vars
        dim = d_model*patch_num

        self.layers = nn.ModuleList([])
        for _ in range(n_vars):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, dim//2),
                    nn.Dropout(head_dp),
                    nn.GELU(),
                    nn.Linear(dim//2, target_len)
                )
            )

    def forward(self, x):

        # Input: 

        # x: bs x nvars x num_patches x d_model

        # Output:

        # out: bs x nvars x target_len

        outs = []
        for i in range(self.n_vars):
            input = x[:, i, :, :]
            input = input.flatten(start_dim=1)
            out = self.layers[i](input)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        return outs
    
class JEPAtchTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_patches = config["ws"] // config["patch_len"]

        if config["revin"]:
            self.revin = RevIN(num_features=config["in_dim"], eps=1e-5, affine=True)
        else:
            self.revin = None

        if not config["scratch"]:
            if config["load_hub"]:
                print("Loading pretrained JEPAtchTST from Hugging Face Hub")
                self.encoder = JEPAtchTSTEncoder.from_pretrained("vilhess/JEPAtchTST")
            else:
                print("Loading JEPAtchTST from local checkpoint")
                self.encoder = JEPAtchTSTEncoder(config)
                checkpoint_path = config["save_path"]
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                self.encoder.load_state_dict(checkpoint)
            self.encoder.requires_grad_(False if config['freeze_encoder'] else True)
        
        else:
            self.encoder = JEPAtchTSTEncoder(config)

        self.head = PredictorHead(
            n_vars=config["in_dim"],
            patch_num=num_patches,
            d_model=config["d_model"],
            target_len=config["target_len"],
            head_dp=config["head_dp"]
        )
        self.head.requires_grad_(True)

    def forward(self, x):
        if self.revin is not None:
            x = self.revin(x, mode='norm')

        _, h = self.encoder(x)

        prediction = self.head(h)
        prediction = prediction.permute(0, 2, 1)
        if self.revin is not None:
            prediction = self.revin(prediction, mode='denorm')
        return prediction
    
class JEPAtchTSTLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = JEPAtchTST(config)
        self.criterion = nn.MSELoss()

        self.l2loss = StreamMSELoss()
        self.l1loss = StreamMAELoss()

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = self.criterion(prediction, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        self.l2loss.update(pred, y)

    def on_validation_epoch_end(self):
        l2loss = self.l2loss.compute()
        self.log("val_l2loss", l2loss, prog_bar=True, on_epoch=True, sync_dist=True)

        self.l2loss.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        self.l2loss.update(pred, y)
        self.l1loss.update(pred, y)
    
    def on_test_epoch_end(self):
        l2loss = self.l2loss.compute()
        self.log("l2loss", l2loss, prog_bar=True)
        self.l2loss.reset()
        l1loss = self.l1loss.compute()
        self.log("l1loss", l1loss, prog_bar=True)
        self.l1loss.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.2, epochs=self.hparams.epochs, max_lr=self.hparams.lr, steps_per_epoch=self.hparams.len_loader)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
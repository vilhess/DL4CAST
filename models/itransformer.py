import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import lightning as L 
from models.metric import StreamMAELoss, StreamMSELoss

class DataEmbeddingInverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbeddingInverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)
    
class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale=scale
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        self.scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe, bshe ->bhls", queries, keys)
        A = self.dropout(torch.softmax(scores * self.scale, dim=-1))
        V = torch.einsum("bhls, bshd->blhd", A, values)
        return V
    
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attn = attention
        self.query_proj = nn.Linear(d_model, d_keys*n_heads)
        self.key_proj = nn.Linear(d_model, d_keys*n_heads)
        self.value_proj = nn.Linear(d_model, d_values*n_heads)
        self.out_proj = nn.Linear(d_values*n_heads, d_model)
        
        self.n_heads = n_heads
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, D, _ = keys.shape
        H = self.n_heads

        queries = self.query_proj(queries).reshape(B, L, H, -1)
        keys = self.key_proj(keys).reshape(B, D, H, -1)
        values = self.value_proj(values).reshape(B, D, H, -1)

        out = self.inner_attn(queries, keys, values, attn_mask)
        out = out.reshape(B, L, -1)
        return self.out_proj(out)
    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or d_model*4
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x+y)
    
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class iTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_norm=True
        self.seq_len=config.ws
        self.target_len = config.target_len

        d_model=config.d_model
        d_ff = config.d_ff if hasattr(config, "d_ff") else 2*d_model
        n_heads=8
        dropout=0.1
        n_layers= config.n_layers if hasattr(config, "n_layers") else 2

        self.enc_embedding = DataEmbeddingInverted(c_in=self.seq_len, d_model=d_model, dropout=dropout)

        self.encoder = Encoder(
            attn_layers=[ EncoderLayer(
                AttentionLayer(
                    FullAttention(attention_dropout=dropout), d_model=d_model, n_heads=n_heads
                ),
                d_model=d_model, d_ff=d_ff, dropout=dropout, activation="relu"
            ) for l in range(n_layers) ], norm_layer=nn.LayerNorm(d_model)
        )

        self.projector = nn.Linear(d_model, self.target_len)

    def forward(self, x):

        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+1e-5)
            x /= stdev

        x = self.enc_embedding(x)
        x = self.encoder(x)
        forecast = self.projector(x)
        forecast = forecast.transpose(1, 2)

        if self.use_norm:
            forecast = forecast*(stdev.repeat(1, self.target_len, 1))
            forecast = forecast + (means.repeat(1, self.target_len, 1))
            
        return forecast
    
class iTransformerLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = iTransformer(config)
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
        self.log("val_l2loss", l2loss, prog_bar=True)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
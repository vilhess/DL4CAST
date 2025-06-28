import torch 
import torch.nn as nn
from einops import rearrange, repeat
import math
import lightning as L 
from models.metric import StreamMAELoss, StreamMSELoss

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr_adjust = {2: learning_rate * 0.5 ** 1, 4: learning_rate * 0.5 ** 2,
                 6: learning_rate * 0.5 ** 3, 8: learning_rate * 0.5 ** 4,
                 10: learning_rate * 0.5 ** 5}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super().__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        bs, ws, dim = x.size()
        x_segment = rearrange(x, "b (seg_num seg_len) d -> (b d seg_num) seg_len", seg_len=self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, "(b d seg_num) d_model -> b d seg_num d_model", b=bs, d=dim)
        return x_embed
    
class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1./math.sqrt(E)
        scores = torch.einsum("blhe, bshe -> bhls", queries, keys)
        A = self.dropout(torch.softmax(scores * scale, dim=-1))
        V = torch.einsum("bhls, bshd -> blhd", A, values)
        return V.contiguous()
    
class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        d_keys = d_model//n_heads
        d_values = d_model//n_heads

        self.inner_attn = FullAttention(attention_dropout=dropout)
        self.query_proj = nn.Linear(d_model, d_keys*n_heads)
        self.key_proj = nn.Linear(d_model, d_keys*n_heads)
        self.value_proj = nn.Linear(d_model, d_values*n_heads)
        self.out_proj = nn.Linear(d_values*n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_proj(queries).reshape(B, L, H, -1)
        keys = self.key_proj(keys).reshape(B, S, H, -1)
        values = self.value_proj(values).reshape(B, S, H, -1)

        out = self.inner_attn(queries, keys, values)

        out = out.view(B, L, -1)
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
    
class TwoStageAttentionLayer(nn.Module):
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4*d_model
        self.time_attn = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        bs = x.size(0)

        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model") # (bs*dim) patch_num d_model
        time_enc = self.time_attn(time_in, time_in, time_in)

        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.mlp1(dim_in))
        dim_in = self.norm2(dim_in) # (bs*dim) patch_num d_model

        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=bs)
        batch_router = repeat(self.router, "seg_num factor d_model -> (repeat seg_num) factor d_model", repeat=bs)

        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send) # (b*patch_num) factor d_model
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer) # (bs*patch_num) ts_d d_model

        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.mlp2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model", b=bs)
        return final_out
    
class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size*d_model, d_model)
        self.norm_layer = norm_layer(d_model*win_size)

    def forward(self, x):
        bs, ts_dim, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num!=0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1) # bs, ts_dim, seg_num // win_size, d_model * win_size
        x = self.norm_layer(x)
        x = self.linear_trans(x) # bs, ts_dim, seg_num // win_size, d_model
        return x

class scale_block(nn.Module):
    def __init__(self, seg_num, win_size, step, d_model, n_heads, d_ff, n_layers, dropout, factor=10):
        super().__init__()
        if win_size>1:
            self.merge_layers = SegMerging(d_model=d_model, win_size=win_size)
            divider = win_size**step
        else:
            self.merge_layers = None
            divider=1

        self.encoder_layers = nn.ModuleList()
        for i in range(n_layers):
            self.encoder_layers.append(
                TwoStageAttentionLayer(seg_num=seg_num//divider, factor=factor, d_model=d_model, 
                                       d_ff=d_ff, n_heads=n_heads, dropout=dropout)
                )
            
    def forward(self, x):
        bs, ts_dim, L, d_model = x.shape

        if self.merge_layers is not None:
            x = self.merge_layers(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, e_blocks, win_size, d_model, n_heads , d_ff, n_layers, dropout, seg_num, factor=10):
        super().__init__()
        self.encode_blocks = nn.ModuleList()
        self.encode_blocks.append(scale_block(seg_num=seg_num, win_size=1, step=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff, 
                                              n_layers=n_layers, dropout=dropout, factor=factor))

        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(seg_num=seg_num, win_size=win_size, step=i, d_model=d_model, n_heads=n_heads, 
                                                  d_ff=d_ff, n_layers=n_layers, dropout=dropout, factor=factor))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)
        return encode_x

class DecoderLayer(nn.Module):
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num=10, factor=10):
        super().__init__()
        self.self_attn = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, d_ff, dropout)
        self.cross_attn = AttentionLayer(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.linear_pred = nn.Linear(d_model, seg_len)
    
    def forward(self, x, cross):
        batch = x.size(0)

        x = self.self_attn(x)

        x = rearrange(x, "b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model")
        cross = rearrange(cross, "b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model")

        tmp = self.cross_attn(x, cross, cross)

        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.mlp1(y)
        dec_output = self.norm2(x+y)

        dec_output = rearrange(dec_output, "(b ts_d) out_seg_num d_model -> b ts_d out_seg_num d_model", b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, "b ts_d out_seg_num seg_len -> b (ts_d out_seg_num) seg_len")

        return dec_output, layer_predict

class Decoder(nn.Module):
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout, out_seg_num=10, factor=10):
        super().__init__()
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(
                DecoderLayer(seg_len=seg_len, d_model=d_model, n_heads=n_heads, d_ff=d_ff, 
                             dropout=dropout,out_seg_num=out_seg_num, factor=factor)
            )
    def forward(self, x, cross):
        final_predict = None
        i=0

        ts_d = x.size(1)
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict # bs, ts_d*out_seg_num, seg_len
            i+=1
        final_predict = rearrange(final_predict, "b (ts_d seg_num) seg_len -> b (seg_num seg_len) ts_d", ts_d=ts_d)
        return final_predict

class CrossFormer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.data_dim = config.in_dim
        self.in_len = config.ws
        self.out_len = config.target_len
        seg_len = config.seg_len

        win_size=2
        factor=10
        d_model=256
        d_ff = 512
        n_heads=4
        e_layers=config.e_layers or 3
        dropout= config.dropout or 0.2

        self.pad_in_len = math.ceil(1. * self.in_len / seg_len) * seg_len # the size for working with the patch size (seg_len)
        self.pad_out_len = math.ceil(1. * self.out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len # so the padding at the beginning to have ws % patch_size

        self.enc_value_embedding = DSW_embedding(seg_len, d_model) # project patch size to d_model
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        self.encoder = Encoder(e_blocks=e_layers, win_size=win_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=1,
                               dropout=dropout, seg_num=(self.pad_in_len // seg_len), factor=factor)
        
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_out_len // seg_len), d_model))

        self.decoder = Decoder(seg_len=seg_len, d_layers=e_layers+1, d_model=d_model, n_heads=n_heads, d_ff=d_ff, 
                               dropout=dropout, out_seg_num=(self.pad_out_len//seg_len), factor=factor)
        
    def forward(self, x):
        bs = x.size(0)

        if self.in_len_add!=0:
            x = torch.cat((x[:, :1 :].expand(-1, self.in_len_add, -1), x), dim=1)
        
        x = self.enc_value_embedding(x)
        x += self.enc_pos_embedding
        x = self.pre_norm(x)

        enc_out = self.encoder(x)
        
        dec_in = repeat(self.dec_pos_embedding, "b ts_d l d -> (repeat b) ts_d l d", repeat=bs)
        predict = self.decoder(dec_in, enc_out)
        return predict[:, :self.out_len, :]

    
class CrossFormerLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = CrossFormer(config)
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
    
    def on_train_epoch_end(self):
        epoch = self.current_epoch + 1
        adjust_learning_rate(self.optimizers(), epoch, self.hparams.lr)
    
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
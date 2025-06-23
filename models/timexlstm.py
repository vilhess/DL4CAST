import torch 
import torch.nn as nn 
import lightning as L
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from models.metric import StreamMAELoss, StreamMSELoss

    
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    

class xLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_dim = config.in_dim 
        self.target_len = config.target_len

        emb_dim = 256

        if config.revin:
            self.revin = RevIN(num_features=config.in_dim)
        else:
            self.revin = None

        cfg_xlstm = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "vanilla",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=config.ws,
            num_blocks=4,
            embedding_dim=emb_dim,
            slstm_at=[1, 3],
        )

        self.init_projector = nn.Conv1d(config.in_dim, emb_dim, kernel_size=1)
        self.xlstm_stack_right = xLSTMBlockStack(cfg_xlstm)
        self.xlstm_stack_left = xLSTMBlockStack(cfg_xlstm)
        self.out_projector = nn.Linear(config.ws * emb_dim, config.target_len*config.in_dim)
    
    def forward(self, x):

        if self.revin is not None:
            x = self.revin(x, mode='norm')

        x = self.init_projector(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.xlstm_stack_right(x) + self.xlstm_stack_left(x.flip(dims=[1])).flip(dims=[1])
        x = x.reshape(x.shape[0], -1)
        x = self.out_projector(x)
        x = x.reshape(x.shape[0], self.target_len, -1)

        if self.revin is not None:
            x = self.revin(x, mode='denorm')
        return x
    
class xLSTMLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = xLSTM(config)

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
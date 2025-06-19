import torch
import torch.nn as nn
import lightning as L
from torch.optim import Optimizer

from models.metric import StreamMAELoss, StreamMSELoss

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) https://openreview.net/pdf?id=cGDAkQo1C0p
    https://github.com/ts-kim/RevIN
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical µstability
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
    

class SpectralNormalizedAttention(nn.Module):
    """
    Pytorch equivalent of SAMformer: https://github.com/romilbert/samformer/blob/main/models/utils/spectral_norm.py
    I think there is an implementation error in the original code. The code is not similar to the original method.
    """
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)

    def _normalize(self, x):
        with torch.no_grad():
            svd = torch.linalg.svd(x, full_matrices=False)
            singuar_values = svd[1]
            max_singular_value = torch.max(singuar_values, dim=-1, keepdim=True).values
            x = x / max_singular_value.unsqueeze(-1)
        return x

    def forward(self, queries, keys, values):
        query = self._normalize(queries) * self.gamma
        key = self._normalize(keys) * self.gamma
        value = self._normalize(values) * self.gamma
        att_score = nn.functional.scaled_dot_product_attention(query, key, value)
        return  att_score
    

class SAM(Optimizer):
    """
    SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization https://arxiv.org/abs/2010.01412
    https://github.com/davda54/sam
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm
    

class SAMformer(nn.Module):
    def __init__(self, n_channels, seq_len, hid_dim, pred_horizon):
        super().__init__()
        self.revin = RevIN(num_features=n_channels, affine=True)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)
        self.spectral_norm = SpectralNormalizedAttention()
        self.linear_forecaster = nn.Linear(seq_len, pred_horizon)

    def forward(self, x):
        x_norm = self.revin(x, mode='norm')
        x_norm = x_norm.permute(0, 2, 1) 
        queries = self.compute_queries(x_norm)
        keys = self.compute_keys(x_norm)
        values = self.compute_values(x_norm)
        att_score = self.spectral_norm(queries, keys, values)
        out = self.linear_forecaster(att_score)
        out = self.revin(out.permute(0, 2, 1), mode='denorm')
        return out
    

class SAMformerLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = SAMformer(config.in_dim, config.ws, 64, config.target_len)
        self.lr = config.lr
        self.criterion = nn.MSELoss()
        self.l2loss = StreamMSELoss()
        self.l1loss = StreamMAELoss()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()

        x, y = batch
        prediction = self.model(x)
        loss = self.criterion(prediction, y)

        self.manual_backward(loss)
        self.log("train_loss1", loss)
        optim.first_step(zero_grad=True)

        prediction = self.model(x)
        loss = self.criterion(prediction, y)

        self.manual_backward(loss)
        self.log("train_loss2", loss)
        optim.second_step(zero_grad=True)
    
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
        optimizer = SAM(self.parameters(), torch.optim.Adam, lr=self.lr, weight_decay=1e-5, rho=0.5)
        return optimizer
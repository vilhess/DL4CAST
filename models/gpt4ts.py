import numpy as np
import torch
import torch.nn as nn
import lightning as L

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from models.metric import StreamMSELoss, StreamMAELoss

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = True
        self.ws = configs.ws
        self.patch_size = 16
        self.pretrain = True
        self.stride = 8
        self.patch_num = (self.ws - self.patch_size) // self.stride + 1
        self.gpt_layers = 3
        self.d_model = 768
        self.freeze=True
        self.target_len = configs.target_len

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if self.is_gpt:
            if self.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True) 
            else:
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        self.in_layer = nn.Linear(self.patch_size, self.d_model)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.target_len)
        
        if self.freeze and self.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
    
class gpt4tsLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = GPT4TS(config, self.device)
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
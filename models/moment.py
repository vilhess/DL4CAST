import torch
import torch.nn as nn
import lightning as L 
from momentfm import MOMENTPipeline
from models.metric import StreamMAELoss, StreamMSELoss

class MomentLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{config.size}", 
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': config.target_len,
                'n_channels': config.in_dim,

                'freeze_encoder': config.freeze_encoder,
                'freeze_embedder': config.freeze_encoder,
                'freeze_head': False
            },
        )
        self.model.init()
        self.criterion = nn.MSELoss()

        self.len_loader = config.len_loader
        self.epochs = config.epochs
        self.lr = config.lr

        self.l2loss = StreamMSELoss()
        self.l1loss = StreamMAELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 1) 
        mask = torch.ones((x.shape[0], x.shape[2]), device=self.device)
        output = self.model(x_enc=x, input_mask=mask).forecast
        prediction = output.permute(0, 2, 1)
        loss = self.criterion(prediction, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 1) 
        mask = torch.ones((x.shape[0], x.shape[2]), device=self.device)
        output = self.model(x_enc=x, input_mask=mask).forecast
        prediction = output.permute(0, 2, 1)
        self.l2loss.update(prediction, y)

    def on_validation_epoch_end(self):
        l2loss = self.l2loss.compute()
        self.log("val_l2loss", l2loss, prog_bar=True)
        self.l2loss.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 1) 
        mask = torch.ones((x.shape[0], x.shape[2]), device=self.device)
        output = self.model(x_enc=x, input_mask=mask).forecast
        prediction = output.permute(0, 2, 1)
        self.l2loss.update(prediction, y)
        self.l1loss.update(prediction, y)
    
    def on_test_epoch_end(self):
        l2loss = self.l2loss.compute()
        self.log("l2loss", l2loss, prog_bar=True)
        self.l2loss.reset()
        l1loss = self.l1loss.compute()
        self.log("l1loss", l1loss, prog_bar=True)
        self.l1loss.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.3, max_lr=self.lr, total_steps=self.len_loader*self.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
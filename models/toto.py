import lightning as L
import torch
from models.toto_api.data.util.dataset import MaskedTimeseries
from models.toto_api.inference.forecaster import TotoForecaster
from models.toto_api.model.toto import Toto
from models.metric import StreamMSELoss, StreamMAELoss

class TotoLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
        toto.compile()
        self.model = TotoForecaster(toto.model)
        self.target_len = config.target_len

        self.l2loss = StreamMSELoss()
        self.l1loss = StreamMAELoss()

    def test_step(self, batch, batch_idx):
        x, y = batch
        print(x.shape, y.shape)
        ctx = x.squeeze(0)
        timestamp_seconds = torch.zeros(ctx.size(0), ctx.size(1)).to(self.device)
        time_interval_seconds = torch.full((ctx.size(0),), 60*15).to(self.device)
        inputs = MaskedTimeseries(
            series=ctx,
            padding_mask=torch.full_like(ctx, True, dtype=torch.bool),
            id_mask=torch.zeros_like(ctx),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )
        forecast = self.model.forecast(
                        inputs,
                        prediction_length=self.target_len,
                        num_samples=256,
                        samples_per_batch=256,
                        use_kv_cache=True,
                    )
        forecast = forecast.quantile(0.5)
        pred = forecast.permute(0, 2, 1)
        print(pred.shape, y.shape)

        self.l2loss.update(pred, y)
        self.l1loss.update(pred, y)
    
    def on_test_epoch_end(self):
        l2loss = self.l2loss.compute()
        self.log("l2loss", l2loss, prog_bar=True)
        self.l2loss.reset()
        l1loss = self.l1loss.compute()
        self.log("l1loss", l1loss, prog_bar=True)
        self.l1loss.reset()
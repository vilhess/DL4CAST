import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime


from dataset import get_dataloaders
from utils import save_results, load_model

@hydra.main(version_base=None, config_path=f"conf", config_name="config")
def main(cfg: DictConfig):

    torch.manual_seed(0)

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")
    OmegaConf.set_struct(cfg, False)

    settings = cfg.settings

    config_dataset = cfg.dataset
    dataset = config_dataset.name

    config_model = cfg.model
    model_name = config_model.name

    config_model_params = cfg.dataset_model

    wandb_logger = WandbLogger(project='DL4CAST', name=f"{model_name}_{dataset}_{settings.target_len}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    wandb_logger.config = OmegaConf.to_container(cfg, resolve=True)

    if settings.univariate:
        config_dataset.in_dim = 1
    
    config_model_params["in_dim"] = config_dataset.in_dim
    config_model_params["target_len"] = settings.target_len

    trainloader, valloader, testloader = get_dataloaders(config_dataset, config_model_params, settings, num_workers=21, persistent_workers=True, pin_memory=True)

    model = load_model(model_name)

    config_model_params["len_loader"] = len(trainloader)

    LitModel = model(config=config_model_params)

    
    early_stop_callback = EarlyStopping(monitor="val_l2loss", mode="min", patience=config_model_params.patience)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_l2loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        filename="best-checkpoint"
    )

    trainer = L.Trainer(max_epochs=config_model_params.epochs, enable_checkpointing=True, log_every_n_steps=1, 
                        accelerator="gpu", devices=1, strategy="auto", fast_dev_run=False, 
                        gradient_clip_val=config_model.max_norm if hasattr(config_model, "max_norm") else 0,
                        callbacks=[early_stop_callback, checkpoint_callback], logger=wandb_logger)
    
    if config_model.training:
        trainer.fit(model=LitModel, train_dataloaders=trainloader, val_dataloaders=valloader)
        best_model_path = checkpoint_callback.best_model_path
        best_model = model.load_from_checkpoint(best_model_path, config=config_model_params)
        
    else:
        best_model = LitModel

    results = trainer.test(model=best_model, dataloaders=testloader)

    mse_loss = results[0]["l2loss"]
    mae_loss = results[0]["l1loss"]

    wandb_logger.log_metrics({"test_mse": mse_loss, "test_mae": mae_loss})

    print(f"Test MSE: {mse_loss}")
    print(f"Test MAE: {mae_loss}")

    folder="univariate" if settings.univariate else "multivariate"
    save_results(filename=f"results/{folder}/mse.json", dataset=dataset, model=model_name, score=mse_loss, 
                 context_horizon=config_model_params.ws, target_horizon=settings.target_len)
    save_results(filename=f"results/{folder}/mae.json", dataset=dataset, model=model_name, score=mae_loss,
                 context_horizon=config_model_params.ws, target_horizon=settings.target_len)
    
    wandb.finish()

if __name__ == "__main__":
    main()
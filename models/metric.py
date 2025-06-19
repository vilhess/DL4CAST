import torch 
import torch.nn.functional as F

class StreamMSELoss:
    def __init__(self):
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets):
        self.predictions.append(preds)
        self.targets.append(targets)
    
    def compute(self):
        self.predictions = torch.cat(self.predictions).detach().cpu()
        self.targets = torch.cat(self.targets).detach().cpu()
        loss = F.mse_loss(self.predictions, self.targets)
        return loss.item()
    
    def reset(self):
        self.predictions = []
        self.targets = []

class StreamMAELoss:
    def __init__(self):
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets):
        self.predictions.append(preds)
        self.targets.append(targets)
    
    def compute(self):
        self.predictions = torch.cat(self.predictions).detach().cpu()
        self.targets = torch.cat(self.targets).detach().cpu()
        loss = F.l1_loss(self.predictions, self.targets)
        return loss.item()
    
    def reset(self):
        self.predictions = []
        self.targets = []
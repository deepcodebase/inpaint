from typing import Dict, Tuple, Any

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from hydra.utils import instantiate


class LitClassifier(LightningModule):

    def __init__(
        self, cfg: Dict[str, Any], **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.model = instantiate(self.cfg.model)
        self.loss = instantiate(self.cfg.loss)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = self.loss(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log('train_acc5', acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = self.loss(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('val_loss', loss_val, on_epoch=True, logger=True)
        self.log('val_acc1', acc1, prog_bar=True, on_epoch=True, logger=True)
        self.log('val_acc5', acc5, on_epoch=True, logger=True)

    @staticmethod
    def __accuracy(output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optim, self.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer)
        return [optimizer], [scheduler]

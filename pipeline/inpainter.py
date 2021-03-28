import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pytorch_lightning.core import LightningModule
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class LitInpainter(LightningModule):

    def __init__(
        self, cfg: Dict[str, Any], **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.model = instantiate(self.cfg.model)
        self.loss = instantiate(self.cfg.loss)

    def forward(self, img_miss, mask):
        return self.model(img_miss, mask)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        img_miss = img * mask
        output, _, _ = self(img_miss, mask)
        loss, loss_detail = self.loss(output, img, mask)

        self.log(
            'train_loss', loss, on_step=True, on_epoch=True)
        self.log(
            'train/mse', loss_detail['reconstruction_loss'],
            on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            'train/percep', loss_detail['perceptual_loss'],
            on_step=True, on_epoch=True)
        self.log(
            'train/style', loss_detail['style_loss'],
            on_step=True, on_epoch=True)
        self.log(
            'train/tv', loss_detail['total_variation_loss'],
            on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img_miss = img * mask
        fulls, alphas, fills = self(img_miss, mask)
        loss, loss_detail = self.loss(fulls, img, mask)

        self.log(
            'val_loss', loss, on_epoch=True, prog_bar=True)
        self.log(
            'val/mse', loss_detail['reconstruction_loss'],
            on_epoch=True, prog_bar=True)
        self.log(
            'val/percep', loss_detail['perceptual_loss'], on_epoch=True)
        self.log(
            'val/style', loss_detail['style_loss'], on_epoch=True)
        self.log(
            'val/tv', loss_detail['total_variation_loss'], on_epoch=True)

        if self.trainer.is_global_zero:
            # To save n_save * n_image_per_batch samples into files every epoch.
            batch_interval = max(
                1, len(self.trainer.datamodule.val_dataloader()) // (
                    self.trainer.world_size * (self.cfg.val_save.n_save - 1)))
            if batch_idx % batch_interval == 0:
                save_dir = Path(f'result/epoch_{self.current_epoch}/')
                save_dir.mkdir(exist_ok=True, parents=True)
                n = self.cfg.val_save.n_image_per_batch
                full, alpha, fill = fulls[0], alphas[0], fills[0]
                save_image(
                    torch.cat((
                        img[:n], img_miss[:n],
                        alpha[:n], fill[:n], full[:n]), dim=0),
                    save_dir / f'{batch_idx:09d}.jpg', nrow=n)
        return loss

    def _log_metrics(self):
        if self.trainer.is_global_zero:
            str_metrics = ''
            for key, val in self.trainer.logged_metrics.items():
                str_metrics += f'\n\t{key}: {val}'
            logger.info(str_metrics)

    def on_validation_end(self):
        self._log_metrics()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optim, self.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer)
        return [optimizer], [scheduler]

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from utils.log import print_config, save_lr_finder


logger = logging.getLogger(__name__)


def init_trainer(cfg):
    cfg_trainer = dict(cfg.pl_trainer)
    if 'logging' in cfg:
        loggers = []
        for _, cfg_log in cfg.logging.items():
            loggers.append(instantiate(cfg_log))
        cfg_trainer['logger'] = loggers
    if cfg.callbacks:
        callbacks = []
        for _, cfg_callback in cfg.callbacks.items():
            callbacks.append(instantiate(cfg_callback))
        cfg_trainer['callbacks'] = callbacks
    if cfg_trainer['accelerator'] == 'ddp' and cfg_trainer['precision'] < 32:
        cfg_trainer['plugins'] = DDPPlugin(find_unused_parameters=False)

    trainer = pl.Trainer(**cfg_trainer)
    return trainer


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:

    print_config(cfg)

    pl._logger.handlers = []
    pl._logger.propagate = True

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    model = instantiate(cfg.pipeline, cfg=cfg, _recursive_=False)
    trainer = init_trainer(cfg)
    datamodule = instantiate(cfg.dataset)

    if cfg.mode == 'find_lr':
        lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
        save_lr_finder(lr_finder)
        logger.info(f"Suggestion: {lr_finder.suggestion()}")
    else:
        trainer.fit(model, datamodule)

        if cfg.run_test:
            trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()

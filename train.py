import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.plugins import DDPPlugin


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:

    pl._logger.handlers = []
    pl._logger.propagate = True

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    model = instantiate(cfg.pipeline, cfg=cfg)
    if 'find_unused_parameters' in cfg:
        trainer = pl.Trainer(
            **cfg.pl_trainer, plugins=DDPPlugin(
                find_unused_parameters=cfg.find_unused_parameters))
    else:
        trainer = pl.Trainer(**cfg.pl_trainer)
    datamodule = instantiate(cfg.dataset)

    trainer.fit(model, datamodule)
    if cfg.run_test:
        trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()

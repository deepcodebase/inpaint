import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:

    pl._logger.handlers = []
    pl._logger.propagate = True

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    model = instantiate(cfg.pipeline, cfg=cfg)

    cfg_trainer = dict(cfg.pl_trainer)
    if cfg.logging:
        loggers = []
        for cfg_log in cfg.logging:
            loggers.append(instantiate(cfg_log))
        cfg_trainer['logger'] = loggers
    if cfg.callbacks:
        callbacks = []
        for cfg_callback in cfg.callbacks:
            callbacks.append(instantiate(cfg_callback))
        cfg_trainer['callbacks'] =callbacks

    trainer = pl.Trainer(**cfg_trainer)
    datamodule = instantiate(cfg.dataset)
    trainer.fit(model, datamodule)

    if cfg.run_test:
        trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()

import os
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

from contextualized.regression.lightning_modules import ContextualizedRegression
from contextualized.data import *


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage) -> None:
        if isinstance(trainer.logger, WandbLogger):
            self.parser.save(
                self.config,
                os.path.join(trainer.logger.experiment.dir, "run-config.yaml"),
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )


def cli_main():
    """
    Entrypoint for mgen command
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_config = os.path.join(base_dir, "configs/defaults.yaml")
    MyLightningCLI(
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"fit": {"default_config_files": [default_config]}},
        auto_configure_optimizers=False,
        save_config_callback=LoggerSaveConfigCallback,
    )


if __name__ == "__main__":
    cli_main()
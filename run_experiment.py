import logging
import pprint

import wandb
# import torch

from ECGDL.utils import setup_logging
from ECGDL.experiment import ExperimentConfig, run_experiment, run_experiment_unsup


# Initiate Logger
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Setup Experiment Config
    config = ExperimentConfig()
    config.experiment_config = "mitbih_downstream"

    # Init logging
    setup_logging(f'./logs/{config.cur_time}.log', "DEBUG")

    # Select config file
    if config.experiment_config == "emotion_ssl":
        config.default_emotion_self_supervised_config()
    elif config.experiment_config == "lvh_downstream":
        config.default_lvh_downstream_config()
    elif config.experiment_config == "mitbih_downstream":
        config.default_mitbih_downstream_config()
    elif config.experiment_config == "MSwKM":
        config.default_MSwKM_config()
    elif config.experiment_config == "unsup_mts":
        config.default_unsup_mts_config()
    elif config.experiment_config == "autoencoder":
        config.default_autoencoder_config()
    elif config.experiment_config == "default":
        config.default_config()
    else:
        logger.critical("No specify config file!")

    logger.info("Experiment Config:\n%s", pprint.pformat(config.to_dict()))

    # Init wandb
    wandb.init(
        entity=config.wandb_repo, project=config.wandb_project,
        name=config.cur_time, group=config.wandb_group,
        dir=config.wandb_dir, config=config.to_dict(),
    )
    wandb.tensorboard.patch(pytorch=True)

    # Run Experiment
    if config.experiment_config in ["emotion_ssl", "autoencoder", "MSwKM", "unsup_mts"]:
        training_history = run_experiment_unsup(config)
    elif config.experiment_config in ["default", "lvh_downstream", "mitbih_downstream"]:
        training_history, test_report = run_experiment(config)
        logger.info("Test Result:\n%s", pprint.pformat(test_report))
    else:
        logger.warning("No matched experiment config!")

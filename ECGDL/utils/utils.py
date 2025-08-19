import os
import time
import logging
import logging.config

from typing import Optional

import torch
import torch.nn as nn

# Initiate Logger
logger = logging.getLogger(__name__)


def setup_logging(log_path: Optional[str] = None, level: str = "DEBUG"):
    handlers_dict = {
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    }

    if log_path is not None:
        safe_dir(log_path, with_filename=True)
        handlers_dict["file_handler"] = {
            "class": "logging.FileHandler",
            "formatter": "full",
            "level": "DEBUG",
            "filename": log_path,
            "encoding": "utf8"
        }

    # Configure logging
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "[ %(asctime)s ] %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            },
            "full": {
                "format": "[ %(asctime)s ] %(levelname)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": handlers_dict,
        "loggers": {
            "ECGDL": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
            "__main__": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
        }
    }

    # Deal with dual log issue
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger().handlers[0].setLevel(logging.WARNING)

    logging.config.dictConfig(config_dict)
    logger.info("Setup Logging!")


def pretty_stream(tqdm_bar):
    tqdm_bar.close()
    time.sleep(1)


def safe_dir(path: str, with_filename: bool = False) -> str:
    dir_path = os.path.dirname(path) if with_filename else path
    if not os.path.exists(dir_path):
        logger.info("Dir %s not exist, creating directory!", dir_path)
        os.makedirs(dir_path)
    return os.path.abspath(path)


def save_model(epoch: int,
               checkpoint_root: str,
               model: nn.Module,
               optimizer: torch.optim.Optimizer,  # type: ignore
               lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]):  # pylint: disable=protected-access

    checkpoint_root = safe_dir(checkpoint_root)
    checkpoint_path = f'{checkpoint_root}/{model.__class__.__name__}_ckpt_ep{epoch:04d}'
    logger.info('Save model at %s!', checkpoint_path)
    save_dict = {
        'epochs': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if lr_scheduler is not None:
        save_dict['lr_scheduler'] = lr_scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)

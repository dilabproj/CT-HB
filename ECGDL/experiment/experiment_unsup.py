import logging

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb
from ECGDL.datasets.dataset import ECGLeadDataset, RandomECGDatasetSubset, ECGDataset
from ECGDL.datasets.dataset import ECGDatasetUnsupSubset
from ECGDL.datasets.mitbih_dataset import MITBIHUnsupDataset
from ECGDL.training import train_model_unsup, train_model_triplet, train_model_unsup_mts
from ECGDL.experiment.config import ExperimentConfig

# Initiate Logger
logger = logging.getLogger(__name__)


def run_experiment_unsup(config: ExperimentConfig):  # pylint: disable=too-many-branches, too-many-statements
    # Check Pytorch Version Before Running
    logger.info('Torch Version: %s', torch.__version__)  # type: ignore
    logger.info('Cuda Version: %s', torch.version.cuda)  # type: ignore

    # Initialize Writer
    writer_dir = f"{config.tensorboard_log_root}/{config.cur_time}/"
    writer = SummaryWriter(log_dir=writer_dir)

    # Initialize Device
    device = torch.device(f"cuda:{str(config.gpu_device_id[0])}")

    preprocess_lead_name = "None" if config.preprocess_lead is None else config.preprocess_lead.__name__
    logger.info('Preprocess Lead: %s', preprocess_lead_name)

    # Initialize Dataset and Split into train/valid/test DataSets
    logger.info('Global Transform:\n%s', config.global_transform)
    logger.info('Pretext Transform:\n%s', config.pretext_transform)
    if config.db_path == "/home/micro/ecg_lvh_data_20200203.db":
        dataset = ECGDataset(
            db_location=config.db_path,
            target_table=config.target_table,
            target_attr=config.target_attr,
            algo_type=config.algo_type,
            num_neg=config.num_neg,
            target_attr_transform=config.target_attr_transform,
            stratify_attr=config.stratify_attr,
            train_test_patient_possible_overlap=config.train_test_patient_possible_overlap,
            mhash_csv_path=config.mhash_csv_path,
            preprocess_lead=config.preprocess_lead,
            transform=config.global_transform,
            name_mapping=config.target_name_map("short"),
            random_seed=config.random_seed)
        logger.info('Train Transform:\n%s', config.train_transform)
        dataset = ECGDatasetUnsupSubset(  # type: ignore
            dataset, config.algo_type, "train", pretext_transform=config.pretext_transform)
    elif config.db_path == "/home/micro/ecg_data_20190519_20191230_20200130.db":
        dataset = ECGLeadDataset(  # type: ignore
            algo_type=config.algo_type,
            db_location=config.db_path,
            target_table=config.target_table,
            num_neg=config.num_neg,
            mhash_csv_path=config.mhash_csv_path,
            preprocess_lead=config.preprocess_lead,
            transform=config.global_transform,
            pretext_transform=config.pretext_transform,
            negative_transform=config.negative_transform)
        if config.random_subset_ratio is not None:
            dataset = RandomECGDatasetSubset(  # type: ignore
                dataset, random_subset_ratio=config.random_subset_ratio, random_seed=-90863)
    elif config.dataset_type == 'mitbih':
        dataset = MITBIHUnsupDataset(  # type: ignore
            db_path=config.db_path,
            num_neg=config.num_neg,
            algo_type=config.algo_type,
            transform=config.global_transform,
            pretext_transform=config.pretext_transform,
            negative_transform=config.negative_transform)
    else:
        logger.warning("No matched dataset!")

    if config.dataset_sampler is not None:
        dataset_sampler = config.dataset_sampler(dataset, **config.dataset_sampler_args)  # type: ignore
        logger.info('Sampler: %s', config.dataset_sampler.__name__)
    else:
        dataset_sampler = None  # type: ignore
        logger.info('Sampler: None')

    # Shuffle must be False for dataset_sampler to work
    if config.dataset_type == "lvh":
        train_loader = DataLoader(dataset, batch_size=config.batch_size, drop_last=True,
                                  sampler=dataset_sampler, shuffle=(dataset_sampler is None),
                                  num_workers=config.dataloader_num_worker)
    elif config.dataset_type == 'mitbih':
        train_loader = DataLoader(dataset, batch_size=config.batch_size, drop_last=True,
                                  sampler=dataset_sampler, shuffle=(dataset_sampler is None),
                                  num_workers=config.dataloader_num_worker)
    else:
        logger.warning("No train loader!")

    if config.algo_type == "unsup_mts":
        model = nn.DataParallel(config.model(  # type: ignore
            in_channels=dataset.variate_cnt, **config.model_args),
            device_ids=config.gpu_device_id).to(device)  # pylint: disable=bad-continuation
    elif config.model is not None:
        model = nn.DataParallel(config.model(  # type: ignore
            n_variate=dataset.variate_cnt, device=device, **config.model_args),
            device_ids=config.gpu_device_id).to(device)  # pylint: disable=bad-continuation
    else:
        logger.critical("Model is not chosen in config!")
        return None
    wandb.watch(model)
    logger.info('Model: %s', model.__class__.__name__)
    # Log total parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info('Model params: %s', pytorch_total_params)
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model params trainable: %s', pytorch_total_params_trainable)

    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_args)

    logger.info("Training Started!")
    if config.algo_type in ['emotion_ssl', 'cdae', 'dae', 'vae']:
        training_history, _ = train_model_unsup(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            writer=writer,
            pretexts=config.pretext_transform,
            num_epochs=config.num_epochs,
            loss_function=config.loss_function,
            lr_scheduler=config.lr_scheduler,
            lr_scheduler_args=config.lr_scheduler_args,
            checkpoint_root=config.checkpoint_root,
        )
    elif config.algo_type == 'unsup_mts':
        training_history, _ = train_model_unsup_mts(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            writer=writer,
            pretexts=config.global_transform,
            num_epochs=config.num_epochs,
            loss_function=config.loss_function,
            lr_scheduler=config.lr_scheduler,
            lr_scheduler_args=config.lr_scheduler_args,
            checkpoint_root=config.checkpoint_root,
        )
    elif config.algo_type == 'MSwKM':
        training_history, _ = train_model_triplet(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            writer=writer,
            num_epochs=config.num_epochs,
            loss_function=config.loss_function,
            lr_scheduler=config.lr_scheduler,
            lr_scheduler_args=config.lr_scheduler_args,
            checkpoint_root=config.checkpoint_root,
        )
    else:
        logger.warning("No matched algorithm type!")
    logger.info("Training Complete!")

    return training_history

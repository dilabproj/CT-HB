import logging

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb
from ECGDL.datasets.dataset import ECGDataset, ECGDatasetSubset, ImbalancedDatasetSampler, RandomECGDatasetSubset
from ECGDL.datasets.mitbih_dataset import MITBIHDataset, RandomMITBIHDatasetSubset
from ECGDL.training.training import train_model
from ECGDL.training.training_linear import train_model_linear
from ECGDL.evaluate.evaluate import evaluate_model
from ECGDL.evaluate.evaluate_linear import evaluate_model_linear
from ECGDL.experiment.config import ExperimentConfig
from ECGDL.training.model_picker import ModelPicker
from ECGDL.utils import safe_dir
from ECGDL.models import CausalCNNEncoder

# Initiate Logger
logger = logging.getLogger(__name__)


def run_experiment(config: ExperimentConfig):  # pylint: disable=too-many-locals,too-many-statements, too-many-branches
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
    if config.dataset_type == "lvh":
        dataset = ECGDataset(
            db_location=config.db_path,
            target_table=config.target_table,
            target_attr=config.target_attr,
            target_attr_transform=config.target_attr_transform,
            stratify_attr=config.stratify_attr,
            train_test_patient_possible_overlap=config.train_test_patient_possible_overlap,
            mhash_csv_path=config.mhash_csv_path,
            preprocess_lead=config.preprocess_lead,
            transform=config.global_transform,
            name_mapping=config.target_name_map("short"),
            random_seed=config.random_seed)
        logger.info('Train Transform:\n%s', config.train_transform)
        train_dataset = ECGDatasetSubset(dataset, "train", transform=config.train_transform)
        if config.train_sampler_ratio is not None:
            logger.info("Sub-sampling training dataset to %s", config.train_sampler_ratio)
            train_dataset = RandomECGDatasetSubset(  # type: ignore
                train_dataset, config.train_sampler_ratio, random_seed=-43447)
        logger.info('Valid Transform:\n%s', config.valid_transform)
        valid_dataset = ECGDatasetSubset(dataset, "valid", transform=config.valid_transform)
        if config.valid_sampler_ratio is not None:
            logger.info("Sub-sampling validation dataset to %s", config.valid_sampler_ratio)
            valid_dataset = RandomECGDatasetSubset(valid_dataset, config.valid_sampler_ratio)  # type: ignore
        logger.info('Test Transform:\n%s', config.test_transform)
        test_dataset = ECGDatasetSubset(dataset, "test", transform=config.test_transform)
        if config.test_sampler_ratio is not None:
            logger.info("Sub-sampling testing dataset to %s", config.test_sampler_ratio)
            test_dataset = RandomECGDatasetSubset(test_dataset, config.test_sampler_ratio)  # type: ignore
    elif config.dataset_type == "mitbih":
        train_dataset = MITBIHDataset(  # type: ignore
            db_path=config.db_path,
            classes=config.classes,
            data_type='train',
            preprocess_lead=config.preprocess_lead,
        )
        if config.train_sampler_ratio is not None:
            logger.info("Sub-sampling training dataset to %s", config.train_sampler_ratio)
            train_dataset = RandomMITBIHDatasetSubset(  # type: ignore
                train_dataset, config.train_sampler_ratio, random_seed=-43447)
        valid_dataset = MITBIHDataset(  # type: ignore
            db_path=config.db_path,
            classes=config.classes,
            data_type='val',
            preprocess_lead=config.preprocess_lead,
        )
        if config.valid_sampler_ratio is not None:
            logger.info("Sub-sampling validation dataset to %s", config.valid_sampler_ratio)
            valid_dataset = RandomMITBIHDatasetSubset(valid_dataset, config.valid_sampler_ratio)  # type: ignore
        test_dataset = MITBIHDataset(  # type: ignore
            db_path=config.db_path,
            classes=config.classes,
            data_type='test',
            preprocess_lead=config.preprocess_lead,
        )
        if config.test_sampler_ratio is not None:
            logger.info("Sub-sampling testing dataset to %s", config.test_sampler_ratio)
            test_dataset = RandomMITBIHDatasetSubset(test_dataset, config.test_sampler_ratio)  # type: ignore
        dataset = train_dataset  # type: ignore
    else:
        logger.warning('No dataset type!')

    # Create train/valid/test DataLoaders
    # Init Imbalance Sampler if Needed
    if config.dataset_sampler is not None:
        dataset_sampler = config.dataset_sampler(train_dataset, **config.dataset_sampler_args)  # type: ignore
        logger.info('Sampler: %s', config.dataset_sampler.__name__)
    else:
        dataset_sampler = None  # type: ignore
        logger.info('Sampler: None')

    # Calculate Class Weight if Needed
    class_weight = None
    if config.class_weight_transformer is not None:
        _, _, d_class_weight = ImbalancedDatasetSampler.get_weight(train_dataset, config.class_weight_transformer)
        logger.info('Class Weight Transformer: %s', config.class_weight_transformer.__name__)
        class_weight = [x[1] for x in sorted(d_class_weight.items())]
        logger.info('Class Weight: %s', class_weight)

    # Shuffle must be False for dataset_sampler to work
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True,
                              sampler=dataset_sampler, shuffle=(dataset_sampler is None),
                              num_workers=config.dataloader_num_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.dataloader_num_worker)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.dataloader_num_worker)

    if config.model is not None:
        if config.model == CausalCNNEncoder:
            model = nn.DataParallel(config.model(  # type: ignore
                in_channels=dataset.variate_cnt, **config.model_args), device_ids=config.gpu_device_id).to(device)
        elif config.algo_type == 'linear':
            model = config.model(  # type: ignore
                n_class=dataset.class_cnt, n_variate=dataset.variate_cnt,
                device=device, **config.model_args).to(device)
        else:
            model = nn.DataParallel(config.model(  # type: ignore
                n_class=dataset.class_cnt, n_variate=dataset.variate_cnt,
                device=device, **config.model_args), device_ids=config.gpu_device_id).to(device)
        wandb.watch(model)
        logger.info('Model: %s', model.__class__.__name__)
        # Log total parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        logger.info('Model params: %s', pytorch_total_params)
        pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Model params trainable: %s', pytorch_total_params_trainable)
    else:
        logger.critical("Model not chosen in config!")
        return None

    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_args)
    lr_scheduler = None if config.lr_scheduler is None else config.lr_scheduler(optimizer, **config.lr_scheduler_args)

    # Initalize model picker
    model_picker = ModelPicker()
    model_picker.add_metrics(config.target_metrics)

    logger.info("Training Started!")
    if config.algo_type == 'linear':
        training_history, total_steps, clf, scaler = train_model_linear(
            model=model,
            train_loader=train_loader,
            device=device,
            dataset_type=config.dataset_type,
        )
        logger.info("Training Complete!")

        logger.info("Testing Started!")
        test_reports = evaluate_model_linear(
            model, test_loader, device, "Testing", total_steps, writer, clf=clf, scaler=scaler)
        logger.info("Testing Complete!")
        if config.checkpoint_root is not None:
            config.checkpoint_root = safe_dir(config.checkpoint_root)
            df = pd.DataFrame.from_dict(test_reports).transpose()
            df.to_csv(f'{config.checkpoint_root}/test_report_linear.csv')
    else:
        training_history, total_steps = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            writer=writer,
            model_picker=model_picker,
            num_epochs=config.num_epochs,
            loss_function=config.loss_function,
            lr_scheduler=lr_scheduler,
            class_weight=class_weight,
            validation_loader=valid_loader,
            checkpoint_root=config.checkpoint_root,
        )
        logger.info("Training Complete!")

        logger.info("Testing Started!")
        test_reports = {}
        for saved_model_state_dict, model_uuid, reasons, reason_str in model_picker.get_best_models():
            if config.model == CausalCNNEncoder:
                model = nn.DataParallel(config.model(  # type: ignore
                    in_channels=dataset.variate_cnt, **config.model_args), device_ids=config.gpu_device_id).to(device)
            else:
                model = nn.DataParallel(config.model(  # type: ignore
                    n_class=dataset.class_cnt, n_variate=dataset.variate_cnt,
                    device=device, **config.model_args), device_ids=config.gpu_device_id).to(device)
            model.load_state_dict(saved_model_state_dict)
            logger.info("Testing %s due to %s!", model_uuid, reason_str)
            test_report, prediction_details = evaluate_model(
                model, test_loader, device, f"Testing:{reason_str}", total_steps,
                writer, config.loss_function)

            # Save prediction result for charts
            y_truth, y_pred, y_pred_prob = prediction_details
            # TODO: Switch pred_prob to save max prob instead of last category prob
            if config.checkpoint_root is not None:
                config.checkpoint_root = safe_dir(config.checkpoint_root)
                pred_result_df = pd.DataFrame(
                    data={'truth': y_truth, 'pred': y_pred, 'pred_prob': y_pred_prob[:, -1]})
                pred_result_df.to_csv(
                    f"{config.checkpoint_root}/model_{model_uuid}_{reason_str}_final_pred_result.csv", index=False)
                test_report_df = pd.DataFrame(test_report).transpose()
                test_report_df.to_csv(f"{config.checkpoint_root}/model_{model_uuid}_{reason_str}_test_report.csv")

            test_reports[model_uuid] = {
                "reasons": reasons,
                "report": test_report,
                "prediction_details": prediction_details,
            }
        logger.info("Testing Complete!")
        if config.checkpoint_root is not None:
            logger.info("Start storing best models!")
            model_picker.store_models(config.checkpoint_root)
            logger.info("Complete storing best models!")

    return training_history, test_reports

import logging  # pylint: disable=too-many-lines

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional, Dict, Tuple, Any, Type, List

import torch
import torch.nn as nn

from ECGDL.const import LEAD_SAMPLING_RATE
from ECGDL.datasets.ecg_data_model import Base, ECGtoK, NAME_MAPPING_MAPPING
from ECGDL.preprocess import preprocess_leads
from ECGDL.preprocess.transform import RandomSelectAnchorPositiveHB, MITBIHRandomSelectThreeHB, MITBIHConcateFullHB
from ECGDL.preprocess.transform import Scaling, Negation, HorizontalFlipping, Permutation, TimeWarping, NoiseAddition
from ECGDL.preprocess.transform import RandomSelectHB, MINMAX_1D

# from ECGDL.models import CRNN
from ECGDL.models import SSSTM
from ECGDL.models import PIRL
from ECGDL.models import CDAE
from ECGDL.models import DAE
from ECGDL.models import VAE
from ECGDL.models import StandfordModel
from ECGDL.models import CausalCNNEncoder
from ECGDL.models import DownstreamModel

# from ECGDL.training import GWLoss
from ECGDL.training import TripletLoss_unsup, TripletLossVaryingLength, MultiSimilarityLoss
# from ECGDL.training import TripletLoss
from ECGDL.datasets import ECGDataset
from ECGDL.datasets import ImbalancedDatasetSampler, RandomSampler
from ECGDL.datasets.ecg_data_model import ECGtoLVH, ECGLeads


# Initiate Logger
logger = logging.getLogger(__name__)


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries
    So the configs can be used when comparing results across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret


def dfac_cur_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def dfac_dataset_optimizer_args():
    return {
        "lr": 1e-3,
    }


def dfac_model_args():
    return {}


def dfac_lr_scheduler_args():
    return {
        'step_size': 3,
        'gamma': 0.1,
    }


def dfac_target_metrics():
    return [
        ('max', 'epoch'),
        ('min', 'valid', 'Loss'),
        ('max', 'valid', 'Macro AVG', 'AUROC'),
    ]


@dataclass
class ExperimentConfig:  # pylint: disable=too-many-instance-attributes

    # Default experiment setting
    experiment_config: Optional[str] = None

    # Data Sources
    db_path: str = "/home/micro/ecg_data_20190519_20191230.db"

    # GPU Device Setting
    gpu_device_id: Tuple[int] = (1,)

    # Logging Related
    cur_time: str = field(default_factory=dfac_cur_time)
    tensorboard_log_root: str = "/home/crystal/ecg_self_supervised/ECGDL_tb/"
    wandb_dir: str = "/home/crystal/ecg_self_supervised/ECGDL_wandb/"

    # WandB setting
    wandb_repo: str = "phes11925"
    wandb_project: str = "ecg_representation_learning"
    wandb_group: str = "test"

    # Set random seed. Set to None to create new Seed
    random_seed: Optional[int] = None

    # Default No Lead Preprocessing Function
    # Eg. ECGDL.preprocess: preprocess_leads
    preprocess_lead: Optional[Callable] = None

    # Transform Function
    global_transform: Optional[Tuple[Callable, ...]] = None
    pretext_transform: Optional[Tuple[Callable, ...]] = None
    negative_transform: Optional[Tuple[Callable, ...]] = None
    train_transform: Optional[Tuple[Callable, ...]] = None
    valid_transform: Optional[Tuple[Callable, ...]] = None
    test_transform: Optional[Tuple[Callable, ...]] = None

    # Default Target Attr
    target_table: Base = ECGtoK
    target_attr: str = "potassium_gp1"

    # MIT_NIH classes
    classes: Tuple[str, str, str] = ('N', 'S', 'V')

    # Default No Transformation of Target Attribute
    # Eg. ECGDL.dataset.dataset: ECGDataset.transform_gp1_only_hyper
    target_attr_transform: Optional[Callable] = None

    # Default mhash selection csv file
    mhash_csv_path: Optional[str] = None

    # Dataset Stratify Attributes
    stratify_attr: Tuple[str, ...] = ('gender', 'EKG_age', 'EKG_K_interhour', 'pair_type')

    # Do we need to consider train/test patient overlapping?
    # ie. Is there more than one record per patient in the dataset?
    train_test_patient_possible_overlap: bool = True

    # Training Related
    batch_size: int = 64
    dataloader_num_worker: int = 32
    num_neg: int = 1

    # Default No Dataset Sampler
    # Eg. ECGDL.dataset.dataset: ImbalancedDatasetSampler
    dataset_sampler: Optional[Type[torch.utils.data.sampler.Sampler]] = None
    dataset_sampler_args: Dict[str, Any] = field(default_factory=dict)
    dataset_type: str = 'emotion_ssl'
    algo_type: str = 'emotion_ssl'
    linear: bool = False

    # Default Cross Entropy loss
    loss_function: Optional[nn.Module] = None

    # Default No Random Data Sampling
    random_subset_ratio: Optional[float] = None
    train_sampler_ratio: Optional[float] = None
    valid_sampler_ratio: Optional[float] = None
    test_sampler_ratio: Optional[float] = None

    # Default No Class Weight
    class_weight_transformer: Optional[Callable] = None

    # Default Don't Select Model
    model: Optional[Type[torch.nn.Module]] = None
    model_args: Dict[str, Any] = field(default_factory=dict)

    # Default model save root
    checkpoint_root_dir: str = '.'
    checkpoint_root: Optional[str] = None

    # Default Select Adam as Optimizer
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam  # type: ignore
    optimizer_args: Dict[str, Any] = field(default_factory=dfac_dataset_optimizer_args)

    # Default adjust learning rate
    lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None  # pylint: disable=protected-access
    lr_scheduler_args: Dict[str, Any] = field(default_factory=dfac_lr_scheduler_args)

    # Target Metrics for Model Picker
    target_metrics: List[Tuple[str, ...]] = field(default_factory=dfac_target_metrics)

    # Set number of epochs to train
    num_epochs: int = 5

    def target_name_map(self, ntype: str = "short") -> Dict[int, str]:
        if self.target_attr_transform is None:
            if self.target_attr in NAME_MAPPING_MAPPING:
                return NAME_MAPPING_MAPPING[self.target_attr][ntype]
            else:
                logger.warning("Target Attribute Not Transformed and target_attr %s not recognized!",
                               self.target_attr)
                return {}
        else:
            if self.target_attr_transform.__name__ in NAME_MAPPING_MAPPING:
                return NAME_MAPPING_MAPPING[self.target_attr_transform.__name__][ntype]
            else:
                logger.warning("Target Attribute Transformed and Name %s not recognized in mapping!",
                               self.target_attr_transform.__name__)
            return {}

    def to_dict(self, expand: bool = True):
        return transform_dict(asdict(self), expand)

    def default_lvh_downstream_config(self):
        self.dataset_type = "lvh"
        self.checkpoint_root_dir = f'{self.checkpoint_root_dir}/models_saved'
        # Setup Data Source (MD5 Checksum: a1af925cf5edca878912c0995d057a63)
        self.db_path = "/home/micro/ecg_lvh_data_20200203.db"

        # Settings for LVH classification
        self.target_table = ECGtoLVH
        self.stratify_attr = ('gender', 'EKG_age', 'his_HTN', 'his_DM', 'his_MI', 'his_HF', 'his_stroke', 'his_CKD')
        self.target_attr = 'LVH_LVmass_level'
        self.target_attr_transform = ECGDataset.transform_lvh_level
        self.train_test_patient_possible_overlap = False
        self.preprocess_lead = preprocess_leads
        self.batch_size = 64
        self.global_transform = (
            RandomSelectHB(hb_cnt=8, sampling_rate=LEAD_SAMPLING_RATE),
        )

        # Setup model types
        unsup_method_type = "MSwKM"  # MSwKM, unsup_mts, emotion_ssl, cdae, dae
        backbone_model = "causalcnn"  # causalcnn, stanford, original

        # Setup downstream type
        self.algo_type = "non-linear"

        # Setup Logging Group
        self.wandb_group = f"lvh_{unsup_method_type}_{backbone_model}_{self.algo_type}"

        # Setup GPU ID
        self.gpu_device_id = [4, 5, 6, 7]

        # Setup Random Seed
        self.random_seed = 666

        # Use ImbalancedDatasetSampler
        self.dataset_sampler = ImbalancedDatasetSampler
        self.dataset_sampler_args = {
            "num_samples": int(100 * self.batch_size) if self.algo_type == 'non-linear' else 25600,
            "weight_function": ImbalancedDatasetSampler.wf_logxdivx
        }

        # Use Class Weight
        self.class_weight_transformer = ImbalancedDatasetSampler.wf_one

        # Use optimizer
        self.optimizer = torch.optim.Adam
        self.optimizer_args['lr'] = 5e-4

        # Set target metrics
        self.target_metrics = [
            ('max', 'valid', 'LVH', 'AUROC'),
        ]

        # Decide Model
        self.model = DownstreamModel
        self.linear = bool(self.algo_type == 'linear')
        kernel_sz = 25
        naming = f"{self.dataset_type}_{unsup_method_type}_{backbone_model}"
        if unsup_method_type == 'MSwKM' and backbone_model == 'stanford':
            self.model_args = {
                "model_name": PIRL,
                "model_path": f'{self.checkpoint_root_dir}/20200412-211719/DataParallel_ckpt_ep0005',
                "model_agrs": {
                    "model_name": StandfordModel,
                    "model_structure": {
                        'feature_layer': [(64, kernel_sz)],
                        'residual': [
                            [(64, kernel_sz), (64, kernel_sz)],
                            [(64, kernel_sz), (64, kernel_sz)],
                            [(128, kernel_sz), (128, kernel_sz)],
                            [(128, kernel_sz), (128, kernel_sz)],
                            [(256, kernel_sz), (256, kernel_sz)],
                            [(256, kernel_sz), (256, kernel_sz)],
                            [(512, kernel_sz), (512, kernel_sz)],
                            [(512, kernel_sz), (512, kernel_sz)],
                        ],
                        'hb_specific_layers': [
                            [512, 128],
                        ]
                    },
                    "dropout_ratio": 0.3,
                    "signal_len": 300,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == 'MSwKM' and backbone_model == 'causalcnn':
            self.model_args = {
                "model_name": PIRL,
                "model_path": f'{self.checkpoint_root_dir}/20200506-110339_{naming}/DataParallel_ckpt_ep0100',
                "model_agrs": {
                    "model_name": CausalCNNEncoder,
                    "model_structure": {
                        "channels": 40,
                        "depth": 10,
                        "out_channels": 320,
                        "kernel_size": kernel_sz,
                        "reduced_size": 160,
                        'hb_specific_layers': [
                            [320, 128],
                        ]
                    },
                    "dropout_ratio": 0.3,
                    "signal_len": 300,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == 'unsup_mts' and backbone_model == 'causalcnn':
            self.model_args = {
                "model_name": CausalCNNEncoder,
                "model_path": f'{self.checkpoint_root_dir}/20200418-132605_{naming}/DataParallel_ckpt_ep0070',
                "model_agrs": {
                    "channels": 40,
                    "depth": 10,
                    "out_channels": 320,
                    "kernel_size": 25,
                    "reduced_size": 160,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == "emotion_ssl" and backbone_model == "original":
            self.model_args = {
                "model_name": SSSTM,
                "model_path": f'{self.checkpoint_root_dir}/20200426-163021_{naming}/DataParallel_ckpt_ep0100',
                "model_agrs": {
                    "model_name": None,
                    "model_structure": {
                        'shared_layers': [
                            ['conv', 64, 32],
                            ['conv', 64, 32],
                            ['maxpool', 8, 2],
                            ['conv', 128, 16],
                            ['conv', 128, 16],
                            ['maxpool', 8, 2],
                            ['conv', 320, 8],
                            ['conv', 320, 8],
                        ],
                        'task_specific_layers': [
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                        ]
                    },
                    "dropout_ratio": 0.3,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == "cdae" and backbone_model == "original":
            self.model_args = {
                "model_name": CDAE,
                "model_path": f'{self.checkpoint_root_dir}/20200521-135532_{naming}/DataParallel_ckpt_ep0100',
                "model_agrs": {
                    "model_name": None,
                    "model_structure": {
                        'encoder': [
                            32, 16, 4,
                        ],
                        'decoder': [
                            16, 32,
                        ]
                    },
                    "signal_len": 300,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == "dae" and backbone_model == "original":
            self.model_args = {
                "model_name": DAE,
                "model_path": f'{self.checkpoint_root_dir}/20200521-181559_{naming}/DataParallel_ckpt_ep0100',
                "model_agrs": {
                    "model_name": None,
                    "model_structure": {
                        'encoder': [
                            300 * 12, 128 * 12, 64 * 12, 32 * 12,
                        ],
                        'decoder': [
                            64 * 12, 128 * 12, 300 * 12,
                        ]
                    },
                    "signal_len": 300,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }

        # Set Number of Epochs
        self.num_epochs = 100

        # Set model save path
        self.checkpoint_root = f'{self.checkpoint_root_dir}/{self.cur_time}_downstream_{self.algo_type}_{naming}'

    def default_mitbih_downstream_config(self):
        self.dataset_type = "mitbih"
        self.checkpoint_root_dir = f'{self.checkpoint_root_dir}/models_saved'

        # Use Preprocessing Function
        self.db_path = './Data/s2s_mitbih_aami_DS1DS2'
        self.batch_size = 64
        self.classes = ['N', 'S', 'V']

        # Setup model types
        unsup_method_type = "MSwKM"  # MSwKM, unsup_mts, emotion_ssl, cdae, dae, vae
        backbone_model = "causalcnn"  # causalcnn, stanford, original

        # Setup downstream type
        self.algo_type = "non-linear"

        # Setup Logging Group
        self.wandb_group = f"mit_bih_{unsup_method_type}_{backbone_model}_{self.algo_type}_finetuning"

        # Setup GPU ID
        self.gpu_device_id = [4, 5, 6, 7]

        # Setup Random Seed
        self.random_seed = 666

        self.train_sampler_ratio = 0.5
        self.valid_sampler_ratio = self.train_sampler_ratio

        # Use ImbalancedDatasetSampler
        self.dataset_sampler = ImbalancedDatasetSampler
        self.dataset_sampler_args = {
            "num_samples": int(100 * self.train_sampler_ratio * self.batch_size) if self.algo_type == 'non-linear' else 25600,
            "weight_function": ImbalancedDatasetSampler.wf_logxdivx
        }
        # Use Class Weight
        self.class_weight_transformer = ImbalancedDatasetSampler.wf_one

        # Use optimizer
        self.optimizer = torch.optim.Adam
        self.optimizer_args['lr'] = 1e-5

        # Set target metrics
        self.target_metrics = [
            ('max', 'valid', 'Accuracy'),
        ]

        # Decide Model
        if unsup_method_type in ['cdae', 'dae']:
            naming = f"{self.dataset_type}_{unsup_method_type}"
        else:
            naming = f"{self.dataset_type}_{unsup_method_type}_{backbone_model}"
        self.model = DownstreamModel
        self.linear = bool(self.algo_type == 'linear')
        kernel_sz = 19
        if unsup_method_type == "MSwKM" and backbone_model == "stanford":
            channel = 40
            self.model_args = {
                "model_name": PIRL,
                "model_path": f'{self.checkpoint_root_dir}/20200422-194319_{naming}/DataParallel_ckpt_ep0100',
                "model_agrs": {
                    "model_name": StandfordModel,
                    "model_structure": {
                        'feature_layer': [(channel, kernel_sz)],
                        'residual': [
                            [(channel, kernel_sz), (channel, kernel_sz)],
                            [(channel, kernel_sz), (channel, kernel_sz)],
                            [(channel * 2, kernel_sz), (channel * 2, kernel_sz)],
                            [(channel * 2, kernel_sz), (channel * 2, kernel_sz)],
                            [(channel * 4, kernel_sz), (channel * 4, kernel_sz)],
                            [(channel * 4, kernel_sz), (channel * 4, kernel_sz)],
                            [(channel * 8, kernel_sz), (channel * 8, kernel_sz)],
                            [(channel * 8, kernel_sz), (channel * 8, kernel_sz)],
                        ],
                        'hb_specific_layers': [
                            [channel * 8, channel * 4],
                        ]
                    },
                    "dropout_ratio": 0.3,
                    "signal_len": 280,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == "MSwKM" and backbone_model == "causalcnn":
            self.model_args = {
                "model_name": PIRL,
                "model_path": f'{self.checkpoint_root_dir}/20200508-164838_{naming}/DataParallel_ckpt_ep0300',
                "model_agrs": {
                    "model_name": CausalCNNEncoder,
                    "model_structure": {
                        "channels": 40,
                        "depth": 10,
                        "out_channels": 320,
                        "kernel_size": kernel_sz,
                        "reduced_size": 160,
                        'hb_specific_layers': [
                            [320, 128],
                        ]
                    },
                    "dropout_ratio": 0.3,
                },
                "linear": self.linear,
                "freeze": False,
                "dataparallel": True
            }
        elif unsup_method_type == "unsup_mts" and backbone_model == "causalcnn":
            self.model_args = {
                "model_name": CausalCNNEncoder,
                "model_path": f'{self.checkpoint_root_dir}/20200422-190713_{naming}/DataParallel_ckpt_ep0100',
                "model_agrs": {
                    "channels": 40,
                    "depth": 10,
                    "out_channels": 320,
                    "kernel_size": 19,
                    "reduced_size": 160,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == "emotion_ssl" and backbone_model == "original":
            self.model_args = {
                "model_name": SSSTM,
                "model_path": f'{self.checkpoint_root_dir}/20200424-1601133_{naming}/DataParallel_ckpt_ep0100',
                "model_agrs": {
                    "model_name": None,
                    "model_structure": {
                        'shared_layers': [
                            ['conv', 64, 32],
                            ['conv', 64, 32],
                            ['maxpool', 8, 2],
                            ['conv', 128, 16],
                            ['conv', 128, 16],
                            ['maxpool', 8, 2],
                            ['conv', 320, 8],
                            ['conv', 320, 8],
                        ],
                        'task_specific_layers': [
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                            [320, 320, 2],
                        ]
                    },
                    "dropout_ratio": 0.3,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == "cdae" and backbone_model == "original":
            self.model_args = {
                "model_name": CDAE,
                "model_path": f'{self.checkpoint_root_dir}/20200520-135445_{naming}/DataParallel_ckpt_ep0040',
                "model_agrs": {
                    "model_name": None,
                    "model_structure": {
                        'encoder': [
                            32, 16, 4,
                        ],
                        'decoder': [
                            16, 32,
                        ]
                    },
                    "signal_len": 280,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == "dae" and backbone_model == "original":
            self.model_args = {
                "model_name": DAE,
                "model_path": f'{self.checkpoint_root_dir}/20200520-150124_{naming}/DataParallel_ckpt_ep0100',
                "model_agrs": {
                    "model_name": None,
                    "model_structure": {
                        'encoder': [
                            280, 128, 64, 32,
                        ],
                        'decoder': [
                            64, 128, 280,
                        ]
                    },
                    "signal_len": 280,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }
        elif unsup_method_type == "vae" and backbone_model == "original":
            self.model_args = {
                "model_name": VAE,
                "model_path": f'{self.checkpoint_root_dir}/20200521-015937_{naming}/DataParallel_ckpt_ep0100',
                "model_agrs": {
                    "model_name": None,
                    "model_structure": {
                    },
                    "signal_len": 280,
                },
                "linear": self.linear,
                "freeze": True,
                "dataparallel": True
            }

        # Set Number of Epochs
        self.num_epochs = 100

        # Set model save path
        self.checkpoint_root = f'{self.checkpoint_root_dir}/{self.cur_time}_downstream_{self.algo_type}_{naming}'

    def default_emotion_self_supervised_config(self):
        self.algo_type = 'emotion_ssl'
        backbone_model = 'original'
        self.dataset_type = "lvh"
        self.checkpoint_root_dir = f'{self.checkpoint_root_dir}/models_saved'

        # Setup Logging Group
        self.wandb_group = f"{self.dataset_type}_{self.algo_type}_{backbone_model}"

        # Setup GPU ID
        self.gpu_device_id = [4, 5, 6, 7]

        # Setup Random Seed
        self.random_seed = 666

        if self.dataset_type == 'lvh':
            self.db_path = "/home/micro/ecg_lvh_data_20200203.db"

            # Set hyperparameters
            self.batch_size = 64
            signal_len = 5000

            # Settings for unsupervised learning
            self.target_table = ECGtoLVH
            self.stratify_attr = (
                'gender', 'EKG_age', 'his_HTN', 'his_DM', 'his_MI', 'his_HF', 'his_stroke', 'his_CKD')
            self.target_attr = 'LVH_LVmass_level'
            self.target_attr_transform = ECGDataset.transform_lvh_level
            self.train_test_patient_possible_overlap = False

            # Use random sampler
            self.dataset_sampler = RandomSampler
            self.dataset_sampler_args = {
                "replacement": True,
                "num_samples": int(100 * self.batch_size)
            }

            # Use Preprocessing Function
            self.preprocess_lead = preprocess_leads

        elif self.dataset_type == 'mitbih':
            self.db_path = './Data/s2s_mitbih_aami_DS1DS2'
            self.batch_size = 64
            signal_len = 280
        else:
            logger.warning("No matched dataset type!")

        # Use optimizer
        self.optimizer = torch.optim.Adam
        self.optimizer_args['lr'] = 1e-4

        self.loss_function = None

        self.model = SSSTM
        naming = f"{self.dataset_type}_{self.algo_type}_{backbone_model}"
        if backbone_model == 'original':
            self.model_args = {
                "model_name": None,
                "model_structure": {
                    'shared_layers': [
                        ['conv', 64, 32],
                        ['conv', 64, 32],
                        ['maxpool', 8, 2],
                        ['conv', 128, 16],
                        ['conv', 128, 16],
                        ['maxpool', 8, 2],
                        ['conv', 320, 8],
                        ['conv', 320, 8],
                    ],
                    'task_specific_layers': [
                        [320, 320, 2],
                        [320, 320, 2],
                        [320, 320, 2],
                        [320, 320, 2],
                        [320, 320, 2],
                        [320, 320, 2],
                        [320, 320, 2],
                    ]
                },
                "dropout_ratio": 0.3,
                "signal_len": signal_len,
            }

        # Set Number of Epochs
        self.num_epochs = 100

        # Set model save path
        self.pretext_transform = (
            NoiseAddition(signal_len),
            Scaling(0.2),
            Negation(-1),
            HorizontalFlipping(),
            Permutation(10),
            TimeWarping(signal_len),
        )

        self.checkpoint_root = f'{self.checkpoint_root_dir}/{self.cur_time}_{naming}'

    def default_autoencoder_config(self):
        self.algo_type = 'dae'  # cdae, dae, vae
        self.dataset_type = 'lvh'

        # Setup Random Seed
        self.random_seed = 666

        if self.dataset_type == 'lvh':
            self.db_path = "/home/micro/ecg_lvh_data_20200203.db"
            # self.db_path = "/home/micro/ecg_data_20190519_20191230_20200130.db"

            # Set hyperparameters
            self.batch_size = 64

            # Settings for unsupervised learning
            if self.db_path == "/home/micro/ecg_lvh_data_20200203.db":
                self.target_table = ECGtoLVH
                self.stratify_attr = (
                    'gender', 'EKG_age', 'his_HTN', 'his_DM', 'his_MI', 'his_HF', 'his_stroke', 'his_CKD')
                self.target_attr = 'LVH_LVmass_level'
                self.target_attr_transform = ECGDataset.transform_lvh_level
                self.train_test_patient_possible_overlap = False
            else:
                self.mhash_csv_path = './Data/mhash_split_hb_lvmass_20200203.csv'
                self.target_table = ECGLeads

            # Use random sampler
            self.dataset_sampler = RandomSampler
            self.dataset_sampler_args = {
                "replacement": True,
                "num_samples": int(100 * self.batch_size)
            }

            # Use Preprocessing Function
            self.preprocess_lead = preprocess_leads
            self.global_transform = (
                RandomSelectHB(hb_cnt=1, sampling_rate=LEAD_SAMPLING_RATE),
            )
            signal_len = 300
            variant_cnt = 12
        elif self.dataset_type == 'mitbih':
            self.db_path = './Data/s2s_mitbih_aami_DS1DS2'
            self.batch_size = 64
            signal_len = 280
            variant_cnt = 1
        else:
            logger.warning("No matched dataset type!")

        # Setup Logging Group
        self.wandb_group = f"{self.dataset_type}_{self.algo_type}"

        # Setup GPU ID
        self.gpu_device_id = [2, 3]

        # Use optimizer
        self.optimizer = torch.optim.Adam
        self.optimizer_args['lr'] = 1e-4

        self.loss_function = nn.MSELoss()

        if self.algo_type == 'cdae':
            self.model = CDAE
            self.model_args = {
                "model_structure": {
                    'encoder': [
                        32, 16, 4,
                    ],
                    'decoder': [
                        16, 32,
                    ]
                },
                "signal_len": signal_len,
            }
        elif self.algo_type == 'dae':
            self.model = DAE
            self.model_args = {
                "model_structure": {
                    'encoder': [
                        signal_len * variant_cnt, 128 * variant_cnt, 64 * variant_cnt, 32 * variant_cnt,
                    ],
                    'decoder': [
                        64 * variant_cnt, 128 * variant_cnt, signal_len * variant_cnt,
                    ]
                },
                "signal_len": signal_len,
            }
        elif self.algo_type == 'vae':
            self.global_transform = MINMAX_1D()
            self.loss_function = None
            self.optimizer_args['amsgrad'] = True
            self.model = VAE
            self.model_args = {
                "model_structure": {
                },
            }
        else:
            logger.warning("No this type of algo type!")

        # Set Number of Epochs
        self.num_epochs = 100

        # Set model save path
        self.checkpoint_root = f'{self.checkpoint_root_dir}/models_saved/{self.cur_time}_{self.wandb_group}'

    def default_MSwKM_config(self):
        self.algo_type = 'MSwKM'
        self.checkpoint_root_dir = f'{self.checkpoint_root_dir}/models_saved'

        # Setup training methods
        backbone_model = "causalcnn"
        self.dataset_type = 'lvh'  # lvh, mitbih
        self.num_neg = 5

        # Setup Logging Group
        self.wandb_group = f"{self.dataset_type}_{self.algo_type}_{backbone_model}"

        # Setup GPU ID
        self.gpu_device_id = [4, 5, 6, 7]

        # Setup Random Seed
        self.random_seed = 666

        if self.dataset_type == 'lvh':
            self.db_path = "/home/micro/ecg_lvh_data_20200203.db"
            # self.db_path = "/home/micro/ecg_data_20190519_20191230_20200130.db"

            # Set hyperparameters
            self.batch_size = 64
            signal_len = 300
            kernel_sz = 25
            channel = 40

            # Settings for unsupervised learning
            if self.db_path == "/home/micro/ecg_lvh_data_20200203.db":
                self.target_table = ECGtoLVH
                self.stratify_attr = (
                    'gender', 'EKG_age', 'his_HTN', 'his_DM', 'his_MI', 'his_HF', 'his_stroke', 'his_CKD')
                self.target_attr = 'LVH_LVmass_level'
                self.target_attr_transform = ECGDataset.transform_lvh_level
                self.train_test_patient_possible_overlap = False
            else:
                self.mhash_csv_path = './Data/mhash_split_hb_lvmass_20200203.csv'
                self.target_table = ECGLeads

            # Use random sampler
            self.dataset_sampler = RandomSampler
            self.dataset_sampler_args = {
                "replacement": True,
                "num_samples": int(100 * self.batch_size)
            }

            # Use Preprocessing Function
            self.preprocess_lead = preprocess_leads

            # Use transformation
            self.global_transform = (
                RandomSelectAnchorPositiveHB(positive_samples=5, sampling_rate=LEAD_SAMPLING_RATE),
            )
        elif self.dataset_type == 'mitbih':
            self.db_path = './Data/s2s_mitbih_aami_DS1DS2'

            self.batch_size = 20

            # Use transformation
            self.global_transform = MITBIHRandomSelectThreeHB(positive_samples=5)

            signal_len = 280
            kernel_sz = 19
        else:
            logger.warning("No matched dataset type!")

        # Use optimizer
        self.optimizer = torch.optim.Adam
        self.optimizer_args['lr'] = 1e-4

        self.loss_function = MultiSimilarityLoss(k=5)
        # self.loss_function = TripletLoss(k=5)

        naming = f"{self.dataset_type}_{self.algo_type}_{backbone_model}"
        if backbone_model == 'stanford':
            self.model = PIRL
            self.model_args = {
                "model_name": StandfordModel,
                "model_structure": {
                    'feature_layer': [(channel, kernel_sz)],
                    'residual': [
                        [(channel, kernel_sz), (channel, kernel_sz)],
                        [(channel, kernel_sz), (channel, kernel_sz)],
                        [(channel * 2, kernel_sz), (channel * 2, kernel_sz)],
                        [(channel * 2, kernel_sz), (channel * 2, kernel_sz)],
                        [(channel * 4, kernel_sz), (channel * 4, kernel_sz)],
                        [(channel * 4, kernel_sz), (channel * 4, kernel_sz)],
                        [(channel * 8, kernel_sz), (channel * 8, kernel_sz)],
                        [(channel * 8, kernel_sz), (channel * 8, kernel_sz)],
                    ],
                    'hb_specific_layers': [
                        [channel * 8, channel * 4],
                    ]
                },
                "dropout_ratio": 0.3,
                "signal_len": signal_len,
            }
        else:
            self.model = PIRL
            self.model_args = {
                "model_name": CausalCNNEncoder,
                "model_structure": {
                    "channels": 40,
                    "depth": 10,
                    "out_channels": 320,
                    "kernel_size": kernel_sz,
                    "reduced_size": 160,
                    'hb_specific_layers': [
                        [320, 128],
                    ]
                },
                "dropout_ratio": 0.3,
                "signal_len": signal_len,
            }

        # Set Number of Epochs
        self.num_epochs = 100 if self.dataset_type == 'lvh' else 300

        # Set model save path
        self.checkpoint_root = f'{self.checkpoint_root_dir}/{self.cur_time}_{naming}'

    def default_unsup_mts_config(self):
        self.algo_type = 'unsup_mts'
        self.checkpoint_root_dir = f'{self.checkpoint_root_dir}/models_saved'

        self.dataset_type = 'mitbih'  # lvh, mitbih

        # Setup Logging Group
        self.wandb_group = f'{self.dataset_type}_{self.algo_type}'

        # Setup GPU ID
        self.gpu_device_id = [4, 5, 6, 7]

        # Setup Random Seed
        self.random_seed = 666

        self.num_neg = 5

        naming = f"{self.dataset_type}_{self.algo_type}_causalcnn"
        if self.dataset_type == 'mitbih':
            self.db_path = './Data/s2s_mitbih_aami_DS1DS2'
            self.batch_size = 20
            # Use transformation
            self.global_transform = MITBIHConcateFullHB(5000)
            kernel_size = 19
            self.loss_function = TripletLossVaryingLength(None, 5, 1)
        elif self.dataset_type == 'lvh':
            # self.db_path = "/home/micro/ecg_lvh_data_20200203.db"
            self.db_path = "/home/micro/ecg_data_20190519_20191230_20200130.db"

            # Set hyperparameters
            self.batch_size = 64
            kernel_size = 19

            # Settings for unsupervised learning
            if self.db_path == "/home/micro/ecg_lvh_data_20200203.db":
                self.target_table = ECGtoLVH
                self.stratify_attr = (
                    'gender', 'EKG_age', 'his_HTN', 'his_DM', 'his_MI', 'his_HF', 'his_stroke', 'his_CKD')
                self.target_attr = 'LVH_LVmass_level'
                self.target_attr_transform = ECGDataset.transform_lvh_level
                self.train_test_patient_possible_overlap = False
            else:
                self.mhash_csv_path = './Data/mhash_split_hb_lvmass.csv'
                self.target_table = ECGLeads

            # Use random sampler
            self.dataset_sampler = RandomSampler
            self.dataset_sampler_args = {
                "replacement": True,
                "num_samples": int(100 * self.batch_size)
            }

            # Use Preprocessing Function
            self.preprocess_lead = preprocess_leads

            self.loss_function = TripletLoss_unsup(None, 5, 1)

        # Use optimizer
        self.optimizer = torch.optim.Adam
        self.optimizer_args['lr'] = 1e-4

        self.model = CausalCNNEncoder
        self.model_args = {
            "channels": 40,
            "depth": 10,
            "out_channels": 320,
            "kernel_size": kernel_size,
            "reduced_size": 160,
        }

        # Set Number of Epochs
        self.num_epochs = 100

        # Set model save path
        self.checkpoint_root = f'{self.checkpoint_root_dir}/{self.cur_time}_{naming}'

    def default_config(self):
        # Setup Logging Group
        self.wandb_group = "supervised_learning"

        # Setup GPU ID
        self.gpu_device_id = [4, 5, 6, 7]

        # Setup Random Seed
        self.random_seed = 666

        self.dataset_type = "mitbih"

        if self.dataset_type == "lvh":
            kernel_size = 25
            # Setup Data Source (MD5 Checksum: a1af925cf5edca878912c0995d057a63)
            # config.db_path = "/home/micro/ecg_k-ckd-lvh_data_20190828.db"
            # Setup Data Source (MD5 Checksum: 1f3586091dadec1b39cfd6713f357407)
            # config.db_path = "/home/micro/ecg_lvh_data_20191016.db"
            # Setup Data Source (MD5 Checksum: 7da8e0dd4d357191a60896ffcb03bcd7)
            # config.db_path = "/home/micro/ecg_lvh_data_20191106.db"
            # Setup Data Source (MD5 Checksum: f1a569ba58cfb00de2ddf3727e643ef7)
            # config.db_path = "/home/micro/ecg_lvh_data_20191216.db"
            self.db_path = "/home/micro/ecg_lvh_data_20200203.db"

            # Set Target Attribute
            # config.target_attr = "potassium_gp1"
            # config.target_attr_transform = ECGDataset.transform_gp1_only_hyper
            # config.target_attr = "potassium_value"
            # config.target_attr_transform = ECGDataset.transform_gpv_50_57

            # Settings for statement classification
            # config.target_attr = "statements"
            # config.target_attr_transform = ECGDataset.transform_statements_sr
            # config.stratify_attr = ('gender', 'EKG_age')

            # Settings for LVH classification
            self.target_table = ECGtoLVH
            # self.stratify_attr = (
            #     'gender', 'EKG_age', 'his_HTN', 'his_DM', 'his_MI', 'his_HF', 'his_stroke', 'his_CKD')
            # self.target_attr = 'LVH_LVmass_level'
            # self.target_attr_transform = ECGDataset.transform_lvh_level
            # self.train_test_patient_possible_overlap = False

            # Settings for statement classification
            self.target_attr = "statements"
            self.target_attr_transform = ECGDataset.transform_statements_1avb
            self.stratify_attr = ('gender', 'EKG_age')

            # Settings for unsupervised learning
            # config.target_table = ECGLeads

            # Use Preprocessing Function
            self.preprocess_lead = preprocess_leads
            self.batch_size = 64

            # Use transformation
            self.global_transform = (
                RandomSelectHB(hb_cnt=8, sampling_rate=LEAD_SAMPLING_RATE),
            )

            # Set target metrics
            self.target_metrics = [
                ('max', 'valid', '1AVB', 'AUROC'),
            ]

        elif self.dataset_type == "mitbih":
            self.db_path = './Data/s2s_mitbih_aami_DS1DS2'
            self.classes = ['N', 'S', 'V']
            self.batch_size = 64
            kernel_size = 19

            # Set target metrics
            self.target_metrics = [
                ('max', 'valid', 'Accuracy'),
            ]

        # Use ImbalancedDatasetSampler
        self.dataset_sampler = ImbalancedDatasetSampler
        self.dataset_sampler_args = {
            "num_samples": int(100 * self.batch_size),
            "weight_function": ImbalancedDatasetSampler.wf_logxdivx
        }
        # Use Class Weight
        self.class_weight_transformer = ImbalancedDatasetSampler.wf_one

        # Devlopment testing only
        # config.dataset_sampler_args["num_samples"] = 5 * config.batch_size
        # config.train_sampler_ratio = 0.01
        # config.valid_sampler_ratio, config.test_sampler_ratio = 0.03, 0.03

        # Use Loss function
        self.loss_function = None
        # self.loss_function = GWLoss()

        # Use optimizer
        self.optimizer = torch.optim.Adam
        self.optimizer_args['lr'] = 1e-4

        # Use optimizer
        # config.optimizer = torch.optim.SGD
        # linear_scaled_lr = 8.0 * config.optimizer_args['lr'] * config.batch_size / 512.0
        # config.optimizer_args = {
        #     'lr': linear_scaled_lr,
        #     'momentum': 0.8,
        #     'weight_decay': 1e-2,
        # }

        # Set learning rate adjust
        # config.lr_scheduler = torch.optim.lr_scheduler.StepLR

        # Decide Model
        # config.model = FCNModel
        # config.model_args = {
        #     "model_structure": [(128, 16, False), (256, 7, False), (128, 3, False)]
        # }

        # config.model = CRNN
        # config.model_args = {
        #     "model_structure": {
        #         'cnn': [(128, 20), (256, 7), (128, 3)],
        #         'rnn': (128, 1, True)
        #     },
        #     'add_attention_layer': True,
        #     'dropout_ratio': 0.0
        # }

        # self.model = StandfordModel
        # kernel_sz = 19
        # self.model_args = {
        #     "model_structure": {
        #         'feature_layer': [(64, kernel_sz)],
        #         'residual': [
        #             [(64, kernel_sz), (64, kernel_sz)],
        #             [(64, kernel_sz), (64, kernel_sz)],
        #             [(128, kernel_sz), (128, kernel_sz)],
        #             [(128, kernel_sz), (128, kernel_sz)],
        #             [(256, kernel_sz), (256, kernel_sz)],
        #             [(256, kernel_sz), (256, kernel_sz)],
        #             [(512, kernel_sz), (512, kernel_sz)],
        #             [(512, kernel_sz), (512, kernel_sz)],
        #         ],
        #     },
        #     "dropout_ratio": 0.3,
        #     "signal_len": 280,
        # }

        # self.model = SDMCNN
        # self.model_args = {
        #     'block_args': [
        #         {'filter_per_layer': 64, 'kernel_sz': [5, 7], 'pool_type': 'avg_pool', 'glu': True},
        #         {'filter_per_layer': 32, 'kernel_sz': [3, 5], 'pool_type': 'avg_pool', 'glu': True},
        #         {'filter_per_layer': 16, 'kernel_sz': [5, 7], 'pool_type': 'avg_pool', 'glu': True},
        #     ],
        #     'dropout_ratio': 0.2,
        # }

        self.model = CausalCNNEncoder
        self.model_args = {
            "channels": 40,
            "depth": 10,
            "out_channels": 3,
            "kernel_size": kernel_size,
            "reduced_size": 160,
        }

        # Set Number of Epochs
        self.num_epochs = 100

        # Set model save path
        self.checkpoint_root = f'{self.checkpoint_root_dir}/models_saved/{self.cur_time}_supervised'

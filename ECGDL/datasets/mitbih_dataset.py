import random
import logging

from typing import Tuple, Optional, Callable

from torch.utils.data import Dataset

import numpy as np

from ECGDL.datasets.ecg_data_model import MIT_BIH_CLASS_MAPPING, read_mitbih
from ECGDL.preprocess.transform import Compose


# Initiate Logger
logger = logging.getLogger(__name__)


class MITBIHUnsupDataset(Dataset):
    def __init__(self,
                 db_path: str,
                 num_neg: int,
                 algo_type: str,
                 transform: Optional[Tuple[Callable, ...]] = None,
                 pretext_transform: Optional[Tuple[Callable, ...]] = None,
                 negative_transform: Optional[Tuple[Callable, ...]] = None):
        # Load MIT-BIH dataset
        self.data = read_mitbih(db_path, unsup=True, personal=bool(algo_type not in [
            'emotion_ssl', 'cdae', 'dae', 'vae']))
        self.variate_cnt = 1
        self.num_neg = num_neg
        self.algo_type = algo_type
        self.transform = transform
        self.pretext_transform = pretext_transform
        self.negative_transform = negative_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  # pylint: disable=inconsistent-return-statements
        lead_data = np.array(self.data[idx], dtype='float32')
        if self.transform is not None:
            lead_data = self.transform(lead_data)

        # Random sample negative samples
        negative_list = random.sample(list(set(range(0, len(self.data))).difference(set([idx]))), self.num_neg)

        if self.algo_type == "unsup_mts":
            neg_lead_data_full = []
            for neg_idx in negative_list:
                if self.transform is not None:
                    neg_lead_data = self.transform(self.data[neg_idx])
                neg_lead_data_full.append(neg_lead_data)

            return lead_data, np.array(neg_lead_data_full, dtype='float32')
        elif self.algo_type == "MSwKM":
            neg_lead_data_full = []
            for neg_idx in negative_list:
                if self.negative_transform is not None:
                    neg_lead_data = Compose(self.negative_transform)(self.data[neg_idx])
                neg_index = random.sample(range(0, len(self.data[neg_idx])), 1)[0]
                neg_lead_data = self.data[neg_idx][neg_index]
                neg_lead_data_full.append(neg_lead_data)

            return lead_data[0], lead_data[1], np.array(neg_lead_data_full, dtype='float32')
        elif self.algo_type == 'emotion_ssl':
            lead_data = lead_data.transpose(1, 0)
            assert self.pretext_transform is not None, "Should specify pretexts!"
            # Random choose transformation
            aux_target = np.random.randint(2, size=len(self.pretext_transform))
            lead_data = Compose(self.pretext_transform, aux_target)(lead_data)
            if np.count_nonzero(aux_target) != 0:
                aux_target = np.append(aux_target, 0)
            else:
                aux_target = np.append(aux_target, 1)
            return lead_data, aux_target
        elif self.algo_type == 'cdae':
            lead_data = lead_data.transpose(1, 0)
            lead_data_noise = lead_data + np.random.normal(0, 0.001, lead_data.shape)
            return np.array(lead_data_noise, dtype='float32'), lead_data
        elif self.algo_type == 'vae':
            lead_data_noise = lead_data + np.random.normal(0, 0.001, lead_data.shape)
            return np.array(lead_data_noise, dtype='float32'), lead_data
        elif self.algo_type == 'dae':
            lead_data = lead_data.transpose(1, 0)
            return lead_data, lead_data
        else:
            logger.info("No dataset!")


class MITBIHDataset(Dataset):
    def __init__(self,
                 db_path: str,
                 classes: Tuple[str, str, str],
                 data_type: str,
                 smote_flag: bool = False,
                 preprocess_lead: Optional[Callable] = None):
        # Load MIT-BIH dataset
        if data_type == 'train':
            self.data, self.label = read_mitbih(db_path, classes=classes, smote_flag=smote_flag, data_type=data_type)
        elif data_type == 'val':
            self.data, self.label = read_mitbih(db_path, classes=classes, smote_flag=smote_flag, data_type=data_type)
        elif data_type == 'test':
            self.data, self.label = read_mitbih(db_path, classes=classes, smote_flag=smote_flag, data_type=data_type)
        else:
            logger.warning("No this data type!")

        logger.info("Dataset Size: %s", len(self.label))
        self.variate_cnt = 1
        self.class_cnt = len(classes)
        self.preprocess_lead = preprocess_lead
        self.cid2name = {v: k for k, v in MIT_BIH_CLASS_MAPPING.items() if k in classes}
        logger.info("cid2name: %s", self.cid2name)

        # Get each label sample size
        logger.info("Number Samples in N: %d", len(np.where(self.label == 'N')[0]))
        logger.info("Number Samples in S: %d", len(np.where(self.label == 'S')[0]))
        logger.info("Number Samples in V: %d", len(np.where(self.label == 'V')[0]))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        lead_data = self.data[idx]
        if self.preprocess_lead is not None:
            processed_lead_data = [self.preprocess_lead(n) for n in lead_data]
            lead_data = np.array(processed_lead_data, dtype='float32')
        label = MIT_BIH_CLASS_MAPPING[self.label[idx]]

        return lead_data, label


class RandomMITBIHDatasetSubset(Dataset):
    def __init__(self,
                 dataset: MITBIHDataset,
                 random_subset_ratio: Optional[float],
                 random_seed: Optional[int] = None):
        self.dataset = dataset
        assert random_subset_ratio is not None, "None subset ratio!"
        self.random_subset_ratio = random_subset_ratio

        # Setup Random State
        self.random_seed = random_seed if random_seed is not None else random.randint(-999999, 999999)
        logger.info("Setting random seed to %s", self.random_seed)
        random.seed(self.random_seed)

        # Get random indices
        self.indices = random.sample(range(len(dataset)), k=round(len(dataset) * self.random_subset_ratio))
        self.label = self.dataset.label[self.indices]

        # Get each label sample size
        logger.info("Number Samples in N: %d", len(np.where(self.label == 'N')[0]))
        logger.info("Number Samples in S: %d", len(np.where(self.label == 'S')[0]))
        logger.info("Number Samples in V: %d", len(np.where(self.label == 'V')[0]))

        self.variate_cnt = self.dataset.variate_cnt
        self.class_cnt = self.dataset.class_cnt
        self.cid2name = self.dataset.cid2name

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

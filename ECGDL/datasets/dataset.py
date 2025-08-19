import os
import sys
import math
import random
import logging

from typing import Tuple, List, Dict, Sequence, Optional, Callable, Union, ClassVar

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prettytable import PrettyTable

from ECGDL.const import LEAD_NAMES, LEAD_LENGTH, LEAD_DROP_LEN
from ECGDL.datasets.mitbih_dataset import MITBIHDataset, RandomMITBIHDatasetSubset
from ECGDL.datasets.ecg_data_model import Base, ECGtoK, ECGtoLVH, ECGLeads, TaintCode
from ECGDL.datasets.ecg_data_model import unpack_leads, unpack_statements, extract_statments
from ECGDL.preprocess.transform import Compose


# Initiate Logger
logger = logging.getLogger(__name__)


class ECGDataset(Dataset):  # pylint: disable=too-many-instance-attributes
    MINOR_STRATIFY_CNT: ClassVar[Dict[str, int]] = {
        "ECGtoK": 20,
        "ECGtoLVH": 20,
    }
    # Stratify Tolerence: [train-test, train-valid], (rtol, atol)
    STRATIFY_TOL: ClassVar[Dict[str, List[Tuple[float, float]]]] = {
        "ECGtoK": [(0.0, 0.01), (0.0, 0.005)],
        "ECGtoLVH": [(0.0, 0.01), (0.0, 0.005)],
    }

    def __init__(self,
                 db_location: str,
                 target_table: Base = ECGtoK,
                 target_attr: str = 'potassium_gp1',
                 algo_type: Optional[str] = None,
                 num_neg: int = 5,
                 target_attr_transform: Optional[Callable] = None,
                 stratify_attr: Sequence[str] = ('gender', 'EKG_age'),
                 train_test_patient_possible_overlap: bool = True,
                 split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),
                 mhash_csv_path: Optional[str] = None,
                 preprocess_lead: Optional[Callable] = None,
                 transform: Optional[Tuple[Callable, ...]] = None,
                 name_mapping: Optional[Dict[int, str]] = None,
                 random_seed: Optional[int] = None):

        # Save DB Location
        self.db_location = db_location

        # Set mhash selection file
        self.mhash_csv_path = mhash_csv_path

        self.algo_type = algo_type
        self.num_neg = num_neg

        # Preprocess functions
        self.preprocess_lead = preprocess_lead
        self.transform = transform

        # Init pid as None to force `connect_to_db` function get new connection
        self.pid = None
        self.session, self.Session, self.engine = None, None, None

        # Set using metadata from `ECGDL.datasets.ecg_data_model`
        self.variate_cnt = len(LEAD_NAMES)
        self.data_len = LEAD_LENGTH - LEAD_DROP_LEN

        # Save Attributes
        self.target_table = target_table
        self.target_attr = target_attr
        self.stratify_attr = list(stratify_attr)
        self.train_test_patient_possible_overlap = train_test_patient_possible_overlap
        self.target_attr_transform = target_attr_transform
        self.name_mapping = name_mapping

        # Set limitations
        self.minor_stratify_cnt = self.MINOR_STRATIFY_CNT[self.target_table.__name__]
        self.stratify_tol = self.STRATIFY_TOL[self.target_table.__name__]

        # Get all mhash with lead data
        self.ecg_data, self.mhash_data, \
            self.data_cnt, self.class_cnt, self.cid2target, self.cid2name, self.stratify_cols = self._get_mhash()
        self.target2cid = {v: k for k, v in self.cid2target.items()}
        self.name2cid = {v: k for k, v in self.cid2name.items()}

        # Separate the mhash
        self.random_seed = random_seed if random_seed is not None else random.randint(-999999, 999999)
        logger.info("Setting random seed to %s", self.random_seed)
        self.split_ratio = split_ratio
        self.split_indices_dict = self._stratify_split()

    def connect_to_db(self):
        if os.getpid() != self.pid:
            # Set pid for this instance
            self.pid = os.getpid()

            # Close Database Connections if there is already a connection
            if self.session is not None:
                # logger.info("PID %s: Closing Existing Session!", self.pid)
                self.session.close()
            if self.engine is not None:
                # logger.info("PID %s: Closing Existing Engine!", self.pid)
                self.engine.dispose()

            # Re-connect to Database
            try:
                self.engine = create_engine(f"sqlite:///{self.db_location}")
                self.Session = sessionmaker(bind=self.engine)
                self.session = self.Session()
            except Exception:
                logger.critical("PID %s: Can't connect to DB at `%s` !", self.pid, self.db_location)
                sys.exit(-1)

            # Log pid and instance using the connection to prevent multiple pid use same instance
            # Currently confused if this will be an issue as same id might be different object instance

            # logger.info("PID %s: Connect to DB at `%s` using instance `%x`!", self.pid, self.db_location, id(self))
            # worker_info = torch.utils.data.get_worker_info()
            # if worker_info is not None:
            #     logger.info("PID %s: Worker info from pytorch - ID: %s/%s, Dataset instance id:%s", self.pid,
            #                 worker_info.id, worker_info.num_workers, hex(id(worker_info.dataset)))

        return self.session

    def _get_mhash(self) -> Tuple[pd.DataFrame, pd.DataFrame, int, int, Dict[int, int], Dict[int, str], List[str]]:  # pylint: disable=too-many-statements, too-many-branches
        # Get Session
        session = self.connect_to_db()

        # Get all possible mhashes that have LeadData with needed stratify_cols
        if self.target_attr in ["statements", "potassium_value"]:
            stratify_cols = self.stratify_attr
        else:
            stratify_cols = self.stratify_attr + [self.target_attr]

        q = session.query(self.target_table.mhash, self.target_table.patient_id, self.target_table.req_no,
                          *[getattr(self.target_table, sc) for sc in stratify_cols])
        q = q.filter(self.target_table.leads.has(ECGLeads.taint_code < TaintCode.DEFAULT_SAFE))

        if self.target_table == ECGtoLVH:
            # Temporary workaround before implementing intermediate selection methods
            # Filter out one pair per patient and heartbeat count >= 8
            q = q.filter(self.target_table.LVH_Single == 1)
            q = q.filter(self.target_table.leads.has(ECGLeads.heartbeat_cnt >= 8))

        ecg_data = pd.read_sql(q.statement, session.bind)

        # Only leave mhash that is not label to be dropped
        if self.mhash_csv_path is not None:
            mhash_split_hb = pd.read_csv(self.mhash_csv_path)
            ecg_data = ecg_data.join(mhash_split_hb.set_index('mhash'), on='mhash')
            ecg_data = ecg_data[ecg_data['drop'] == 0].reset_index()

        if self.target_attr == "statements":
            q_mhash = ecg_data.mhash.tolist()
            q = session.query(ECGLeads.mhash, ECGLeads.statements).filter(ECGLeads.mhash.in_(q_mhash))
            ecglead_data = pd.read_sql(q.statement, session.bind)
            logger.debug(ecg_data.head())
            logger.debug(ecglead_data.head())
            ecg_data = ecg_data.join(ecglead_data.set_index('mhash'), on='mhash')
            logger.debug(ecg_data.head())

        target_cnt = ecg_data.groupby([self.target_attr]).count()["mhash"].reset_index()
        logger.info("Target Count:\n%s", target_cnt)

        # Transform Target Column if target_attr_transform exist
        if self.target_attr_transform is not None:
            ecg_data[f"{self.target_attr}_transformed"] = ecg_data[self.target_attr].map(self.target_attr_transform)
            self.target_attr = f"{self.target_attr}_transformed"
            target_cnt = ecg_data.groupby([self.target_attr]).count()["mhash"].reset_index()
            logger.info("Transformed Target Count:\n%s", target_cnt)

        # Drop Unwanted Data
        ecg_data = ecg_data[ecg_data[self.target_attr] != -1].reset_index()
        target_cnt = ecg_data.groupby([self.target_attr]).count()["mhash"].reset_index()
        logger.info("Drop Target Count:\n%s", target_cnt)

        # Need to map to 0 ~ n class id
        cid2target = target_cnt.to_dict()[self.target_attr]
        logger.info("cid2target: %s", cid2target)

        class_cnt = len(cid2target.keys())
        logger.info("Class Count: %s", class_cnt)
        if class_cnt < 2:
            logger.warning("Less than two class found!")

        # Map the name of the class
        cid2name = cid2target
        if self.name_mapping is not None:
            cid2name = {k: self.name_mapping[v] for k, v in cid2name.items()}
        logger.info("cid2name: %s", cid2name)

        # Clean and Bin the columns for stratify split
        if 'EKG_age' in self.stratify_attr:
            ecg_data['EKG_age'] = ecg_data['EKG_age'].fillna(value=-5)
            age_bins = [-10, -1, 20, 30, 40, 50, 60, 70, 80, 200]
            # Map back to 0 ~ k
            ecg_data['EKG_age'] = np.digitize(ecg_data['EKG_age'], age_bins) - 3
        if 'EKG_K_interhour' in self.stratify_attr:
            ecg_data['EKG_K_interhour'] = ecg_data['EKG_K_interhour'].fillna(value=-5)
            interhour_bins = [-10, -1, 3, 6, 9, 12, 15, 18, 21, 24]
            ecg_data['EKG_K_interhour'] = np.digitize(ecg_data['EKG_K_interhour'], interhour_bins)

        # Map pair_type 3 to pair_type 2
        if 'pair_type' in self.stratify_attr:
            ecg_data['pair_type'] = ecg_data['pair_type'].map({1: 1, 2: 2, 3: 2})

        # Create strat_id column
        stratify_cols = self.stratify_attr + [self.target_attr]
        ecg_data['strat_id'] = '#'
        for col in stratify_cols:
            ecg_data['strat_id'] = ecg_data['strat_id'] + ecg_data[col].astype(str).values + "#"

        # Group by stratify id
        count_groups = ecg_data.groupby('strat_id').count()

        # Take out minor group which number smaller than MINOR_STRATIFY_COUNT
        minor_index = count_groups['mhash'][count_groups['mhash'] < self.minor_stratify_cnt].index
        logger.info("Number of minor stratify groups (instance count < %s): %s with %s instances",
                    self.minor_stratify_cnt, len(minor_index), len(ecg_data[ecg_data['strat_id'].isin(minor_index)]))
        minor_tar, minor_tar_cnt = np.unique([i.split("#")[-2] for i in minor_index], return_counts=True)
        minor_tar_df = pd.DataFrame(data={self.target_attr: minor_tar, 'cnt': minor_tar_cnt})
        logger.info("Minor Target Distribution:\n%s", minor_tar_df)

        # Switch Minor strat_id back to target
        # For some target_attr with many different values, change to a single class and assume random
        if self.target_attr in ["statements", "potassium_value"]:
            ecg_data.loc[ecg_data['strat_id'].isin(minor_index), 'strat_id'] = "MINOR_RANDOM"
        else:
            ecg_data.loc[ecg_data['strat_id'].isin(minor_index), 'strat_id'] = ecg_data[self.target_attr].astype(str)

        # Save only mhash and strat_id
        mhash_data = ecg_data[['mhash', 'patient_id', 'req_no', 'strat_id', *stratify_cols]]

        # Get Total Data Count
        data_cnt = len(mhash_data)
        logger.info("Data Count: %s", data_cnt)
        if data_cnt == 0:
            logger.warning("No data Found!")

        return ecg_data, mhash_data, data_cnt, class_cnt, cid2target, cid2name, stratify_cols

    def _stratify_split(self) -> Dict[str, List[int]]:  # pylint: disable=too-many-locals
        train_ratio, valid_ratio, test_ratio = self.split_ratio

        # Setup Random State
        random.seed(self.random_seed)

        if self.train_test_patient_possible_overlap:
            # Split the testing data without patient overlapping
            unique_patient_id = np.unique(self.mhash_data['patient_id'])
            testing_patient_id = random.sample(unique_patient_id.tolist(), k=round(test_ratio * len(unique_patient_id)))
            test_id = self.mhash_data[self.mhash_data['patient_id'].isin(testing_patient_id)].index
            train_valid_id = self.mhash_data[~self.mhash_data['patient_id'].isin(testing_patient_id)].index

            # Check that no patient_id overlaps
            patient_ids = self.mhash_data['patient_id']
            assert set(patient_ids.iloc[test_id]) & set(patient_ids.iloc[train_valid_id]) == set()
        else:
            # Stratify Split the testing data without considering patient overlapping
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=self.random_seed)
            train_valid_id, test_id = next(sss.split(self.mhash_data['mhash'], self.mhash_data['strat_id']))

            # Warns if patient_id overlaps
            patient_ids = self.mhash_data['patient_id']
            if set(patient_ids.iloc[test_id]) & set(patient_ids.iloc[train_valid_id]) != set():
                logger.warning("Setting set to not consider train/test patient possible overlap but patient overlaps!")

        validintrain_ratio = valid_ratio / (train_ratio + valid_ratio)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validintrain_ratio, random_state=self.random_seed)
        train_id, valid_id = next(sss.split(
            self.mhash_data['mhash'].iloc[train_valid_id], self.mhash_data['strat_id'].iloc[train_valid_id]))

        # Transform back to train_valid_id index
        train_id = train_valid_id[train_id]
        valid_id = train_valid_id[valid_id]

        # Double Check that there isn't overlapping indexes
        assert set(train_id) | set(valid_id) | set(test_id) == set(range(self.data_cnt))
        assert set(train_id) & set(valid_id) == set()
        assert set(train_id) & set(test_id) == set()
        assert set(valid_id) & set(test_id) == set()

        # Check on Stratify Quality
        stratify_cols = self.stratify_attr + [self.target_attr]
        for col in stratify_cols:
            logger.info("Stratifying Column: %s", col)

            # Get the distribution for certain stratify_attr
            u_train, cnt_train = np.unique(self.ecg_data[col].iloc[train_id], return_counts=True)
            u_valid, cnt_valid = np.unique(self.ecg_data[col].iloc[valid_id], return_counts=True)
            u_test, cnt_test = np.unique(self.ecg_data[col].iloc[test_id], return_counts=True)
            d_train = cnt_train / sum(cnt_train)
            d_valid = cnt_valid / sum(cnt_valid)
            d_test = cnt_test / sum(cnt_test)

            # Check that attribute list is the same
            assert set(u_train) == set(u_valid) and set(u_valid) == set(u_test), \
                (set(u_train), set(u_valid), set(u_test))

            tb = PrettyTable()
            tb.field_names = ["Type", "Training", "Validation", "Testing"]
            for name, dn, dv, dt, cn, cv, ct in zip(u_train, d_train, d_valid, d_test, cnt_train, cnt_valid, cnt_test):
                tb.add_row([name, f"{cn} ({dn:.5f})", f"{cv} ({dv:.5f})", f"{ct} ({dt:.5f})"])
            tb.align = "r"
            logger.info("Split Distribution: %s\n%s", col, tb.get_string())

            # Check that distribution is similar
            large_tol, small_tol = self.stratify_tol
            assert np.allclose(d_train, d_valid, *small_tol), np.isclose(d_train, d_valid, *small_tol)
            assert np.allclose(d_train, d_test, *large_tol), np.isclose(d_train, d_test, *large_tol)
            assert np.allclose(d_valid, d_test, *large_tol), np.isclose(d_valid, d_test, *large_tol)

        return {
            "train": train_id,
            "valid": valid_id,
            "test": test_id,
        }

    @staticmethod
    def transform_gp1_only_hyper(x: int) -> int:
        return 1 if x == 3 else 0

    @staticmethod
    def transform_gpv_50_57(x: float) -> int:
        if x > 5.7:
            return 1
        elif x < 5:
            return 0
        return -1

    @staticmethod
    def transform_statements_sr(x: str) -> int:
        statements = unpack_statements(x)
        all_safe = []
        for statement in statements:
            sid, sname, _exp = statement
            sname = sname.upper()
            all_safe.append(sid == "SR" or sname == "SINUS RHYTHM")
            if all_safe[-1] != all_safe[0]:
                return -1
        return 0 if all(all_safe) else 1

    @staticmethod
    def transform_statements_1avb(x: str) -> int:
        extracted_statements = extract_statments(x)
        targeted_statement_task = "1AVB#FIRST DEGREE AV BLOCK"
        return 1 if targeted_statement_task in extracted_statements else 0

    @staticmethod
    def transform_lvh_level(x: int) -> int:
        # Skip unknown/test data
        if x == -1:
            return -1
        return 1 if x >= 2 else 0

    def __len__(self):
        return self.data_cnt

    def __getitem__(self, idx):
        # Get Session
        session = self.connect_to_db()

        # Select the targeted mhash
        item = session.query(self.target_table).filter(
            self.target_table.mhash == self.mhash_data["mhash"].iloc[int(idx)]).one()

        # Unpack Lead data and remove last LEAD_DROP_LEN
        lead_data_dict = unpack_leads(item.leads.lead_data)

        if self.preprocess_lead is not None:
            lead_data_processed = [self.preprocess_lead(lead_data_dict[n][:-LEAD_DROP_LEN]) for n in LEAD_NAMES]
        else:
            lead_data_processed = [lead_data_dict[n][:-LEAD_DROP_LEN] for n in LEAD_NAMES]
        lead_data = np.array(lead_data_processed, dtype='float32')

        if self.transform is not None:
            lead_data = Compose(self.transform)(lead_data)

        # Get target value and map to cid
        category = self.target2cid[self.mhash_data[self.target_attr].iloc[int(idx)]]

        if self.algo_type in ["MSwKM", "unsup_mts"]:
            # Random sample negative samples
            negative_list = random.sample(list(set(range(0, self.data_cnt)).difference(set([idx]))), self.num_neg)

            neg_lead_data_full = []
            for neg_idx in negative_list:
                # Select the targeted mhash
                item = session.query(self.target_table).filter(
                    self.target_table.mhash == self.ecg_data["mhash"].iloc[int(neg_idx)]).one()

                # Unpack Lead data and remove last LEAD_DROP_LEN
                neg_lead_data = unpack_leads(item.leads.lead_data)

                if self.preprocess_lead is not None:
                    neg_lead_data_processed = [self.preprocess_lead(
                        neg_lead_data[n][:-LEAD_DROP_LEN]) for n in LEAD_NAMES]
                else:
                    neg_lead_data_processed = [neg_lead_data[n][:-LEAD_DROP_LEN] for n in LEAD_NAMES]

                neg_lead_data = np.array(neg_lead_data_processed, dtype='float32')

                if self.transform is not None:
                    # This is for random select HB
                    neg_lead_data = Compose(self.transform)(neg_lead_data)[0]
                neg_lead_data_full.append(neg_lead_data)

            neg_lead_data = np.array(neg_lead_data_full)
            return lead_data, neg_lead_data
        elif self.algo_type == "cdae":
            lead_data_noise = lead_data + np.random.normal(0, 0.001, lead_data.shape)
            return np.array(lead_data_noise, dtype='float32'), lead_data
        elif self.algo_type == "dae":
            return lead_data, lead_data
        else:
            # Get Patient Details
            # patient_uuids = ['mhash', 'patient_id', 'req_no']
            # patient_details = self.mhash_data[[*patient_uuids, *self.stratify_cols]].iloc[int(idx)].to_dict()

            return lead_data, category


class ECGLeadDataset(Dataset):
    def __init__(self,
                 algo_type: str,
                 db_location: str,
                 mhash_csv_path: Optional[str] = None,
                 target_table: Base = ECGLeads,
                 num_neg: int = 5,
                 preprocess_lead: Optional[Callable] = None,
                 transform: Optional[Tuple[Callable, ...]] = None,
                 pretext_transform: Optional[Tuple[Callable, ...]] = None,
                 negative_transform: Optional[Tuple[Callable, ...]] = None):

        # Save DB Location
        self.db_location = db_location

        # Set type of dataset
        self.algo_type = algo_type

        # Set mhash selection file
        assert mhash_csv_path is not None, "Should specify mhash data!"
        self.mhash_csv_path = mhash_csv_path

        # Preprocess functions
        self.preprocess_lead = preprocess_lead
        self.transform = transform
        self.pretext_transform = pretext_transform
        self.negative_transform = negative_transform

        # Init pid as None to force `connect_to_db` function get new connection
        self.pid = None
        self.session, self.Session, self.engine = None, None, None

        # Set using metadata from `ECGDL.datasets.ecg_data_model`
        self.variate_cnt = len(LEAD_NAMES)
        self.data_len = LEAD_LENGTH - LEAD_DROP_LEN

        # Save Attributes
        self.target_table = target_table

        # Get all mhash with lead data
        self.num_neg = num_neg
        self.ecg_data, self.data_cnt = self._get_mhash()
        self.mhash_data = self.ecg_data

    def connect_to_db(self):
        if os.getpid() != self.pid:
            # Set pid for this instance
            self.pid = os.getpid()

            # Close Database Connections if there is already a connection
            if self.session is not None:
                # logger.info("PID %s: Closing Existing Session!", self.pid)
                self.session.close()
            if self.engine is not None:
                # logger.info("PID %s: Closing Existing Engine!", self.pid)
                self.engine.dispose()

            # Re-connect to Database
            try:
                self.engine = create_engine(f"sqlite:///{self.db_location}")
                self.Session = sessionmaker(bind=self.engine)
                self.session = self.Session()
            except Exception:
                logger.critical("PID %s: Can't connect to DB at `%s` !", self.pid, self.db_location)
                sys.exit(-1)

        return self.session

    def _get_mhash(self) -> Tuple[pd.DataFrame, int]:
        # Get Session
        session = self.connect_to_db()

        q = session.query(self.target_table.mhash, self.target_table.patient_id, self.target_table.req_no)
        q = q.filter(self.target_table.taint_code < TaintCode.DEFAULT_SAFE)
        q = q.filter(self.target_table.heartbeat_cnt >= 8)

        ecg_data = pd.read_sql(q.statement, session.bind)
        logger.info("Before join %s\n", len(ecg_data))

        # Only leave mhash that is not label to be dropped
        logger.info("Load mhash csv file from path %s\n", self.mhash_csv_path)
        mhash_lvh = pd.read_csv(self.mhash_csv_path)
        logger.info("Length of mhash %s\n", len(mhash_lvh))
        ecg_data = ecg_data.join(mhash_lvh.set_index('mhash'), on='mhash')
        ecg_data = ecg_data[ecg_data['drop'].isna()].reset_index()

        # Get Total Data Count
        data_cnt = len(ecg_data)
        logger.info("Data Count: %s", data_cnt)
        if data_cnt == 0:
            logger.warning("No data Found!")

        return ecg_data, data_cnt

    def _get_negative_samples(self, idx, session, take_one: bool = True):
        # Random sample negative samples
        negative_list = random.sample(list(set(range(0, self.data_cnt)).difference(set([idx]))), self.num_neg)

        neg_lead_data_full = []
        for neg_idx in negative_list:
            # Select the targeted mhash
            item = session.query(self.target_table).filter(
                self.target_table.mhash == self.ecg_data["mhash"].iloc[int(neg_idx)]).one()

            # Unpack Lead data and remove last LEAD_DROP_LEN
            neg_lead_data = unpack_leads(item.lead_data)

            if self.preprocess_lead is not None:
                neg_lead_data_processed = [self.preprocess_lead(
                    neg_lead_data[n][:-LEAD_DROP_LEN]) for n in LEAD_NAMES]
            else:
                neg_lead_data_processed = [neg_lead_data[n][:-LEAD_DROP_LEN] for n in LEAD_NAMES]

            neg_lead_data_processed = np.array(neg_lead_data_processed, dtype='float32')

            if self.transform is not None:
                neg_lead_data_processed = Compose(self.transform)(neg_lead_data_processed)
            if take_one:
                neg_lead_data_processed = neg_lead_data_processed[0]
            neg_lead_data_full.append(neg_lead_data_processed)
        return np.array(neg_lead_data_full, dtype='float32')

    def __len__(self):
        return self.data_cnt

    def __getitem__(self, idx):  # pylint: disable=inconsistent-return-statements
        # Get Session
        session = self.connect_to_db()

        # Select the targeted mhash
        item = session.query(self.target_table).filter(
            self.target_table.mhash == self.ecg_data["mhash"].iloc[int(idx)]).one()

        # Unpack Lead data and remove last LEAD_DROP_LEN
        lead_data_dict = unpack_leads(item.lead_data)

        if self.preprocess_lead is not None:
            lead_data_processed = [self.preprocess_lead(lead_data_dict[n][:-LEAD_DROP_LEN]) for n in LEAD_NAMES]
        else:
            lead_data_processed = [lead_data_dict[n][:-LEAD_DROP_LEN] for n in LEAD_NAMES]
        lead_data = np.array(lead_data_processed, dtype='float32')

        if self.transform is not None:
            lead_data = Compose(self.transform)(lead_data)

        if self.algo_type == 'emotion_ssl':
            assert self.pretext_transform is not None, "Should specify pretexts!"
            # Random choose transformation
            aux_target = np.random.randint(2, size=len(self.pretext_transform))
            lead_data = Compose(self.pretext_transform, aux_target)(lead_data)
            return lead_data, aux_target
        elif self.algo_type == 'ssae':
            return lead_data, lead_data
        elif self.algo_type == 'pirl':
            assert self.pretext_transform is not None, "Should specify pretexts!"
            pos_lead_data = Compose(self.pretext_transform)(lead_data)
            neg_lead_data = self._get_negative_samples(idx, session, take_one=False)
            return lead_data, pos_lead_data, neg_lead_data
        elif self.algo_type == "MSwKM":
            neg_lead_data = self._get_negative_samples(idx, session, take_one=True)
            return lead_data[0], lead_data[1], neg_lead_data
        elif self.algo_type == "unsup_mts":
            neg_lead_data = self._get_negative_samples(idx, session, take_one=True)
            return lead_data, neg_lead_data
        elif self.algo_type == "cdae":
            lead_data_noise = lead_data + np.random.normal(0, 0.001, lead_data.shape)
            return np.array(lead_data_noise, dtype='float32'), lead_data
        else:
            logger.warning("No dataset_type Found!")

    @staticmethod
    def my_collate(batch):
        anchor_data = [item[0] for item in batch]
        positive_data = [item[1] for item in batch]
        negative_data = []
        hb_list = []
        for item in batch:
            for hb in item[2]:
                negative_data.append(hb)
            hb_list.append(len(item[2]))
        return torch.tensor(anchor_data), torch.tensor(positive_data), torch.tensor(negative_data)  # pylint: disable=not-callable


class ECGDatasetSubset(Dataset):
    def __init__(self,
                 dataset: ECGDataset,
                 data_type: str,
                 transform: Optional[Tuple[Callable, ...]] = None):
        self.dataset = dataset
        self.data_type = data_type
        assert data_type in ["train", "valid", "test"]
        self.transform = transform
        self.indices = dataset.split_indices_dict[data_type]
        self.mhash_data: pd.DataFrame = self.dataset.mhash_data.iloc[self.indices]

        # Copy Parent Information
        self.variate_cnt: int = self.dataset.variate_cnt
        self.data_len: int = self.dataset.data_len
        self.target_table: Base = self.dataset.target_table
        self.target_attr: str = self.dataset.target_attr
        self.stratify_attr: Sequence[str] = self.dataset.stratify_attr
        self.random_seed: int = self.dataset.random_seed
        self.target2cid: Dict[int, int] = self.dataset.target2cid
        self.cid2target: Dict[int, int] = self.dataset.cid2target
        self.class_cnt: int = self.dataset.class_cnt
        self.cid2name: Dict[int, str] = self.dataset.cid2name

    def __getitem__(self, idx):
        lead_data, category = self.dataset[self.indices[idx]]
        if self.transform is not None:
            lead_data = Compose(self.transform)(lead_data)
        return lead_data, category

    def __len__(self):
        return len(self.indices)


class RandomECGDatasetSubset(Dataset):
    def __init__(self,
                 dataset: Union[ECGDataset, ECGLeadDataset, ECGDatasetSubset],
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
        self.mhash_data: pd.DataFrame = self.dataset.mhash_data.iloc[self.indices]

        # Copy Parent Information
        self.variate_cnt: int = self.dataset.variate_cnt
        self.data_len: int = self.dataset.data_len
        self.target_table: Base = self.dataset.target_table
        if self.dataset.__class__.__name__ != "ECGLeadDataset":
            self.target_attr: str = self.dataset.target_attr  # type: ignore
            self.stratify_attr: Sequence[str] = self.dataset.stratify_attr  # type: ignore
            self.origdataset_random_seed: int = self.dataset.random_seed  # type: ignore
            self.target2cid: Dict[int, int] = self.dataset.target2cid  # type: ignore
            self.cid2target: Dict[int, int] = self.dataset.cid2target  # type: ignore
            self.class_cnt: int = self.dataset.class_cnt  # type: ignore
            self.cid2name: Dict[int, str] = self.dataset.cid2name  # type: ignore

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ECGDatasetUnsupSubset(Dataset):
    def __init__(self,
                 dataset: ECGDataset,
                 algo_type: str,
                 data_type: str,
                 pretext_transform: Optional[Tuple[Callable, ...]] = None):
        self.dataset = dataset
        self.algo_type = algo_type
        self.data_type = data_type
        assert data_type in ["train", "valid", "test"]
        self.pretext_transform = pretext_transform
        self.indices = dataset.split_indices_dict[data_type]
        self.mhash_data: pd.DataFrame = self.dataset.mhash_data.iloc[self.indices]

        # Copy Parent Information
        self.variate_cnt: int = self.dataset.variate_cnt
        self.data_len: int = self.dataset.data_len
        self.target_table: Base = self.dataset.target_table
        self.target_attr: str = self.dataset.target_attr
        self.stratify_attr: Sequence[str] = self.dataset.stratify_attr
        self.random_seed: int = self.dataset.random_seed
        self.target2cid: Dict[int, int] = self.dataset.target2cid
        self.cid2target: Dict[int, int] = self.dataset.cid2target
        self.class_cnt: int = self.dataset.class_cnt
        self.cid2name: Dict[int, str] = self.dataset.cid2name

    def __getitem__(self, idx):
        lead_data, category = self.dataset[self.indices[idx]]
        if lead_data.shape[0] == 1:
            lead_data = lead_data.reshape((self.variate_cnt, -1))
        if self.algo_type == 'emotion_ssl':
            assert self.pretext_transform is not None, "Should specify pretexts!"
            # Random choose transformation
            aux_target = np.random.randint(2, size=len(self.pretext_transform))
            lead_data = Compose(self.pretext_transform, aux_target)(lead_data)
            if np.count_nonzero(aux_target) != 0:
                aux_target = np.append(aux_target, 0)
            else:
                aux_target = np.append(aux_target, 1)
            return lead_data, aux_target
        elif self.algo_type == 'pirl':
            if category.shape[0] == 1:
                category = category.reshape((self.variate_cnt, -1))
            pos_lead_data = Compose(self.pretext_transform)(lead_data[1])
            return lead_data[0], pos_lead_data, category
        elif self.algo_type == 'MSwKM':
            return lead_data[0], lead_data[1], category
        else:
            return lead_data, category

    def __len__(self):
        return len(self.indices)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self,  # pylint: disable=super-init-not-called
                 dataset: Union[
                     ECGDataset, ECGDatasetSubset, RandomECGDatasetSubset, MITBIHDataset, RandomMITBIHDatasetSubset],
                 weight_function: Callable,
                 num_samples: Optional[int] = None):
        self.dataset = dataset
        self.weight_function = weight_function

        # Get distribution of target
        self.data_uni, self.data_cnt, self.weight_mapping = self.get_weight(self.dataset, self.weight_function)
        logger.info("Weight Mapping: %s", self.weight_mapping)

        # Set weight for each sample
        if isinstance(dataset, (MITBIHDataset, RandomMITBIHDatasetSubset)):
            targets = np.array(self.dataset.label)  # type: ignore
        elif isinstance(dataset, (ECGDataset, ECGDatasetSubset, RandomECGDatasetSubset)):
            targets = self.dataset.mhash_data[self.dataset.target_attr]  # type: ignore
        else:
            logger.warning("No this type of dataset!")
        self.weights = torch.tensor([self.weight_mapping[tar] for tar in targets], dtype=torch.float64)  # pylint: disable=not-callable

        # Save number of samples per iteration
        self.num_samples = num_samples if num_samples is not None else len(self.dataset)
        logger.info("Weighted to total %s samples", self.num_samples)

    @staticmethod
    def get_weight(dataset: Union[ECGDataset, ECGDatasetSubset, RandomECGDatasetSubset,
                                  MITBIHDataset, RandomMITBIHDatasetSubset],
                   weight_function: Callable):
        if isinstance(dataset, (MITBIHDataset, RandomMITBIHDatasetSubset)):
            targets = np.array(dataset.label)
        elif isinstance(dataset, (ECGDataset, ECGDatasetSubset, RandomECGDatasetSubset)):
            targets = dataset.mhash_data[dataset.target_attr]
        else:
            logger.warning("No this type of dataset!")
        data_uni, data_cnt = np.unique(targets, return_counts=True)
        weight_mapping = {u: weight_function(c) for u, c in zip(data_uni, data_cnt)}
        return data_uni, data_cnt, weight_mapping

    @staticmethod
    def wf_one(x):  # pylint: disable=unused-argument
        return 1.0

    @staticmethod
    def wf_x(x):
        return x

    @staticmethod
    def wf_onedivx(x):
        return 1.0 / x

    @staticmethod
    def wf_logxdivx(x):
        return math.log(x) / x

    @staticmethod
    def wf_onedivlog(x):
        return 1 / math.log(x)

    def __iter__(self):
        return (i for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class RandomSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, replacement=False, num_samples=None):  # pylint: disable=super-init-not-called
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples

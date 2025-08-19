import re
import json
import math
import enum
import zlib
import hashlib
import logging

from struct import pack, unpack
from typing import Optional, Dict, List, Tuple, Any, Set

import scipy.io

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Binary, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy_utils.types.choice import ChoiceType

from imblearn.over_sampling import SMOTE

from ECGDL.const import LEAD_NAMES, LEAD_LENGTH, LEAD_SAMPLING_RATE, LEAD_DROP_LEN

from ECGDL.preprocess.transform import RandomSelectHB
from ECGDL.preprocess.signal_preprocess import preprocess_leads

# Initiate Logger
logger = logging.getLogger(__name__)


def tablerepr(self):
    repr_dat = []
    for k in sorted(self.__dict__.keys()):
        if k[0] != '_':
            if isinstance(self.__dict__[k], bytes):
                repr_dat.append(f"{k}=<binary>")
            else:
                repr_dat.append(f"{k}={repr(self.__dict__[k])}")
    return "<{}({})>".format(self.__class__.__name__, ', '.join(repr_dat))


Base = declarative_base()  # type: Any
Base.__repr__ = tablerepr

Base_Legacy = declarative_base()  # type: Any
Base_Legacy.__repr__ = tablerepr

K_GP1_NAME_MAPPING = {
    "full": {
        1: "Hypokalemia < 3.5",
        2: "Normal 3.5-5.0",
        3: "Hyperkalemia > 5.0",
    },
    "short": {
        1: "Hypo",
        2: "Normal",
        3: "Hyper",
    },
}

K_GP2_NAME_MAPPING = {
    "full": {
        1: "Severe hypokalemia < 2.5",
        2: "Moderate hypokalemia 2.5-3.0",
        3: "Mild hypokalemia 3.0-3.5",
        4: "Normal 3.5-5.0",
        5: "Mild hyperkalemia 5.0-5.5",
        6: "Moderate hyperkalemia 5.5-6.0",
        7: "Severe hyperkalemia > 6.0",
    },
    "short": {
        1: "Severe Hypo",
        2: "Moderate Hypo",
        3: "Mild hypo",
        4: "Normal",
        5: "Mild Hyper",
        6: "Moderate Hyper",
        7: "Severe Hyper",
    },
}

K_GP3_NAME_MAPPING = {
    "full": {
        1: "Severe hypokalemia < 2.5",
        2: "Moderate hypokalemia 2.5-3.0",
        3: "Mild hypokalemia 3.0-3.5",
        4: "Low Normal 3.5-4.0",
        5: "Mid Normal 4.0-4.5",
        6: "High Normal 4.5-5.0",
        7: "Mild hyperkalemia 5.0-5.5",
        8: "Moderate hyperkalemia 5.5-6.0",
        9: "Severe hyperkalemia > 6.0",
    },
    "short": {
        1: "Severe Hypo",
        2: "Moderate Hypo",
        3: "Mild Hypo",
        4: "Low Normal",
        5: "Mid Normal",
        6: "High Normal",
        7: "Mild Hyper",
        8: "Moderate Hyper",
        9: "Severe Hyper",
    },
}

K_TRANSGP1_ONLY_HYPER_NAME_MAPPING = {
    "full": {
        0: "Not Hyperkalemia < 5.0",
        1: "Hyperkalemia > 5.0",
    },
    "short": {
        0: "Not Hyper",
        1: "Hyper",
    },
}

K_TRANSGPV_50_57_NAME_MAPPING = {
    "full": {
        0: "Not Hyperkalemia < 5.0",
        1: "Hyperkalemia > 5.7",
    },
    "short": {
        0: "Not Hyper",
        1: "Hyper",
    },
}

K_TRANSST_SR_NAME_MAPPING = {
    "full": {
        0: "Sinus Rhythm",
        1: "Not Sinus Rhythm",
    },
    "short": {
        0: "SR",
        1: "Not SR",
    },
}

K_TRANSST_1AVB_NAME_MAPPING = {
    "full": {
        0: "1AVB",
        1: "Not 1AVB",
    },
    "short": {
        0: "1AVB",
        1: "Not 1AVB",
    },
}

LVH_NAME_MAPPING = {
    "full": {
        0: "Not Left Ventricular Hypertrophy",
        1: "Left Ventricular Hypertrophy",
    },
    "short": {
        0: "Not LVH",
        1: "LVH",
    },
}

MIT_BIH_CLASS_MAPPING = {
    'N': 0,
    'S': 1,
    'V': 2,
    'F': 3,
    'Q': 4,
}

NAME_MAPPING_MAPPING = {
    "potassium_gp1": K_GP1_NAME_MAPPING,
    "potassium_gp2": K_GP2_NAME_MAPPING,
    "potassium_gp3": K_GP3_NAME_MAPPING,
    "transform_gp1_only_hyper": K_TRANSGP1_ONLY_HYPER_NAME_MAPPING,
    "transform_gpv_50_57": K_TRANSGPV_50_57_NAME_MAPPING,
    "transform_statements_sr": K_TRANSST_SR_NAME_MAPPING,
    "transform_statements_1avb": K_TRANSST_1AVB_NAME_MAPPING,
    # "LVH_LVmass": LVH_NAME_MAPPING,
    # "LVH_LVmass_BSA": LVH_NAME_MAPPING,
    "transform_lvh_level": LVH_NAME_MAPPING,
}

LeadData = Dict[str, List[float]]
StatementData = List[Tuple[str, ...]]


class TaintCode(enum.Enum):
    # Larger Value Indicate More Serious Problems
    NORMAL = 0
    STRANGE_XMLHASH = 105
    IGN_DUPLICATE_XMLHASH = 204
    IGN_DUPLICATE_MHASH = 205
    DEFAULT_SAFE = 300
    EMPTY_LEAD = 305
    STRANGE_STATEMENT = 402
    STRANGE_REPORTDT = 403
    STRANGE_PATIENTID = 404
    STRANGE_REQNO = 405
    DUPLICATE_XMLHASH = 504
    DUPLICATE_MHASH = 505


TaintData = List[Tuple[TaintCode, Dict[str, str]]]
TaintLabel = Tuple[TaintCode, TaintData]


class ECGtoK(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "ECGtoK"
    id = Column(Integer, primary_key=True)
    mhash = Column(String(16), ForeignKey('ECGLeads.mhash'), unique=True)
    leads = relationship("ECGLeads", backref="kdata", uselist=False)
    req_no = Column(String(16))
    patient_id = Column(String(16))
    gender = Column(Integer)
    EKG_age = Column(Integer)
    EKG_K_interhour = Column(Float)
    theshortest = Column(Integer)
    pair_type = Column(Integer)
    potassium_value = Column(Float)
    potassium_gp1 = Column(Integer)
    potassium_gp2 = Column(Integer)
    potassium_gp3 = Column(Integer)
    his_CKD_MDRD = Column(Integer)
    his_CKD = Column(Integer)
    HR = Column(Integer)
    PR = Column(Integer)
    QRS = Column(Integer)
    QT = Column(Integer)
    QTc = Column(Integer)
    P = Column(Integer)
    T = Column(Integer)
    RR = Column(Integer)
    QRSd = Column(Integer)


# 1105 ECGtoLVH table definition
# class ECGtoLVH(Base):
#     __tablename__ = "ECGtoLVH"
#     id = Column(Integer, primary_key=True)
#     mhash = Column(String(16), ForeignKey('ECGLeads.mhash'), unique=True)
#     leads = relationship("ECGLeads", backref="lvhdata", uselist=False)
#     req_no = Column(String(16))
#     patient_id = Column(String(16))
#     gender = Column(Integer)
#     EKG_age = Column(Integer)
#     LVH_Testing = Column(Integer)
#     LVH_OriginalSingle = Column(Integer)
#     LVH_LVmass_BSA = Column(Integer)
#     LVH_LVmass_BSA_raw = Column(Float)
#     LVH_LVmass_BSA_level = Column(Integer)
#     LVH_LVmass = Column(Integer)
#     LVH_IVSd_PWT = Column(Integer)


# 0114 ECGtoLVH table definition
class ECGtoLVH(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "ECGtoLVH"
    id = Column(Integer, primary_key=True)
    mhash = Column(String(16), ForeignKey('ECGLeads.mhash'), unique=True)
    leads = relationship("ECGLeads", backref="lvhdata", uselist=False)
    req_no = Column(String(16))
    patient_id = Column(String(16))
    gender = Column(Integer)
    EKG_age = Column(Integer)
    echo_req_no = Column(String(16))
    # Patient Disease History
    his_HTN = Column(Integer)
    his_DM = Column(Integer)
    his_MI = Column(Integer)
    his_HF = Column(Integer)
    his_stroke = Column(Integer)
    his_CKD = Column(Integer)
    # Is this data testing data
    LVH_Testing = Column(Integer)
    # We choose one LVH label per patient
    LVH_Single = Column(Integer)
    # This can be used for pretraining
    # LVH_IVSd_PWT = Column(Integer)
    # Raw value for LVmass
    LVH_LVmass_raw = Column(Float)
    LVH_LVmass_level = Column(Integer)
    # Doctor Annotation
    LVH_ECG = Column(Integer)
    # Date for Echo and ECG
    Echo_Date = Column(DateTime)
    ECG_Date = Column(DateTime)
    # Weight and Height for Echo and ECG
    # Weight_Echo_Date = Column(DateTime)
    # Height_Echo_Date = Column(DateTime)
    # Weight_Echo = Column(Float)
    # Height_Echo = Column(Float)
    # BSA_Echo = Column(Float)
    # Weight_ECG_Date = Column(DateTime)
    # Height_ECG_Date = Column(DateTime)
    # Weight_ECG = Column(Float)
    # Height_ECG = Column(Float)
    # BSA_ECG = Column(Float)
    # Raw value for LVmass_BSA
    # LVH_LVmass_BSA_Echo_raw = Column(Float)
    # LVH_LVmass_BSA_Echo_level = Column(Integer)
    # LVH_LVmass_BSA_ECG_raw = Column(Float)
    # LVH_LVmass_BSA_ECG_level = Column(Integer)
    # LVH_LVmass_BSA_raw = Column(Float)
    # LVH_LVmass_BSA_level = Column(Integer)


class ECGLeads_Legacy(Base_Legacy):  # pylint: disable=too-few-public-methods
    __tablename__ = "ECGLeads"
    id = Column(Integer, primary_key=True)
    mhash = Column(String(16), unique=True)
    req_no = Column(String(16))
    patient_id = Column(String(16))  # No zeros in XML
    xmlhash = Column(String(32), unique=True)
    lead_data = Column(Binary)
    statements = Column(Text)
    report_dt = Column(DateTime)
    taint_code = Column(ChoiceType(TaintCode, impl=Integer()), default=TaintCode.NORMAL)
    taint_history = Column(Text, default="")


class ECGLeads(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "ECGLeads"
    id = Column(Integer, primary_key=True)
    mhash = Column(String(16), unique=True)
    req_no = Column(String(16))
    patient_id = Column(String(16))  # No zeros in XML
    xmlhash = Column(String(32), unique=True)
    lead_data = Column(Binary)
    statements = Column(Text)
    report_dt = Column(DateTime)
    taint_code = Column(ChoiceType(TaintCode, impl=Integer()), default=TaintCode.NORMAL)
    taint_history = Column(Text, default="")
    heartbeat_cnt = Column(Integer)


def get_mhash(req_no: str, patient_id: str, salt: Optional[str] = None) -> str:
    s = hashlib.sha256()
    s.update(f"{req_no}{salt}{patient_id}".encode("utf-8"))
    return s.hexdigest()


def pack_leads(lead_data: LeadData) -> bytes:
    assert sorted(lead_data.keys()) == sorted(LEAD_NAMES)
    packed_data: List[float] = []
    for lead_name in LEAD_NAMES:
        assert len(lead_data[lead_name]) == LEAD_LENGTH
        packed_data += lead_data[lead_name]
    return zlib.compress(pack('d' * LEAD_LENGTH * len(LEAD_NAMES), *packed_data))


def unpack_leads(pack_lead_data: bytes) -> LeadData:
    f_lead_data = unpack('d' * LEAD_LENGTH * len(LEAD_NAMES), zlib.decompress(pack_lead_data))
    assert len(f_lead_data) == LEAD_LENGTH * len(LEAD_NAMES)
    ret = {}
    for lead_id, lead_name in enumerate(LEAD_NAMES):
        data_loc = lead_id * LEAD_LENGTH
        ret[lead_name] = list(f_lead_data[data_loc:data_loc + LEAD_LENGTH])
    return ret


def pack_statements(statement_data: StatementData) -> str:
    packed_statements = []
    for statement in statement_data:
        clean_statement = []
        for s in statement:
            assert s.find("^#^") == -1 and s.find("^$^") == -1
            clean_statement.append(s.strip().replace("\n", ""))
        packed_statements.append("^$^".join(clean_statement))
    return "^#^".join(packed_statements)


def unpack_statements(pack_statement_data: str) -> StatementData:
    statements = pack_statement_data.split("^#^")
    return [tuple(statement.split("^$^")) for statement in statements]


def pack_taint_history(taint_history: TaintData) -> str:
    packed_taint_history = []
    for taint in taint_history:
        taint_attr = [taint[0].name]
        for k, v in taint[1].items():
            kv = f"{k}:{v}"
            assert kv.find("#") == -1 and kv.find("$") == -1 and len(kv.split(":")) == 2
            taint_attr.append(kv)
        packed_taint_history.append("$".join(taint_attr))
    return "#".join(packed_taint_history)


def unpack_taint_history(pack_taint_data: str) -> TaintData:
    taint_history = pack_taint_data.split("#")
    ret = []
    for taint in taint_history:
        if taint != "":
            taint_dat = taint.split("$")
            taint_dict = {attr.split(":")[0]: attr.split(":")[1] for attr in taint_dat[1:]}
            ret.append((TaintCode[taint_dat[0]], taint_dict))
    return ret


def create_db(db_loc: str, drop_tables: bool = False):
    engine = create_engine(db_loc)
    if drop_tables:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Done creating DB!")
    engine.dispose()


def move_lead_by_mhash(src_db_session, dst_db_session, mhash: str):
    src_xml_item = src_db_session.query(ECGLeads_Legacy).filter(ECGLeads_Legacy.mhash == mhash).first()
    dst_xml_item = dst_db_session.query(ECGLeads).filter(ECGLeads.mhash == mhash).first()
    if src_xml_item is not None and dst_xml_item is None:
        # Calculate the heartbeat_cnt for this ecg data
        # Unpack Lead data, remove last LEAD_DROP_LEN, and preprocess lead data
        lead_data_dict = unpack_leads(src_xml_item.lead_data)
        lead_data_processed = [preprocess_leads(lead_data_dict[n][:-LEAD_DROP_LEN]) for n in LEAD_NAMES]
        hb_counter = RandomSelectHB(hb_cnt=-1, sampling_rate=LEAD_SAMPLING_RATE, extract_length=(0.2, 0.4))
        heartbeat_cnt = len(hb_counter(np.array(lead_data_processed, dtype='float32')))

        xml_dat = {
            "mhash": src_xml_item.mhash,
            "req_no": src_xml_item.req_no,
            "patient_id": src_xml_item.patient_id,
            "xmlhash": src_xml_item.xmlhash,
            "lead_data": src_xml_item.lead_data,
            "statements": src_xml_item.statements,
            "report_dt": src_xml_item.report_dt,
            "taint_code": src_xml_item.taint_code,
            "taint_history": src_xml_item.taint_history,
            "heartbeat_cnt": heartbeat_cnt,
        }
        # Add them to new DB
        dst_db_session.add(ECGLeads(**xml_dat))
        dst_db_session.commit()


def add_k_ckd_data(k_data_loc: str, ckd_data_loc: str, src_db_session, dst_db_session):
    # Read K Data
    k_data = pd.read_csv(k_data_loc)
    # Read CKD Data
    ckd_data = pd.read_csv(ckd_data_loc).fillna(-1)

    for _index, row in tqdm(k_data.iterrows(), total=len(k_data)):
        # Map K_Data to new Format
        row_dat = {
            "req_no": row["ReqNo"],
            "patient_id": row["patientid_his"],
        }
        row_dat["mhash"] = get_mhash(row_dat["req_no"], row_dat["patient_id"], "#")

        assert row["gender"] in ["M", "F"]
        row_dat["gender"] = 1 if row["gender"] == "M" else 0

        row_dat["EKG_age"] = int(row["EKG_age"])
        row_dat["EKG_K_interhour"] = float(row["EKG_K_interhour"])
        row_dat["theshortest"] = int(row["theshortest"])

        assert row["pair_type"] in ["EKG before K", "K before EKG", "at the same time"]
        if row["pair_type"] == "EKG before K":
            row_dat["pair_type"] = 1
        elif row["pair_type"] == "K before EKG":
            row_dat["pair_type"] = 2
        else:
            row_dat["pair_type"] = 3

        for basic_info in ['HR', 'PR', 'QRS', 'QT', 'QTc', 'P', 'T', 'RR', 'QRSd']:
            row_dat[basic_info] = int(row[basic_info])

        row_dat["potassium_value"] = float(row["potassium_value"])

        for gp in ['potassium_gp1', 'potassium_gp2', 'potassium_gp3']:
            row_dat[gp] = int(row[gp].split(":")[0])

        row_ckd = ckd_data[(ckd_data["ReqNo"] == row["ReqNo"]) & (ckd_data["patientid_his"] == row["patientid_his"])]
        if len(row_ckd) == 1:
            r_ckd = row_ckd.iloc[0]
            assert r_ckd["his_CKD"] in [-1, 0, 1]
            row_dat["his_CKD"] = int(r_ckd['his_CKD'])  # Don't know why it will fail if no type tansform
            assert r_ckd["his_CKD_MDRD"] in [-1, 1, 2, 3, 4, 5]
            row_dat["his_CKD_MDRD"] = int(r_ckd['his_CKD_MDRD'])
            if int(r_ckd['k_hyper'].split(":")[0]) != row_dat['potassium_gp1']:
                print("Mismatch K!", row["ReqNo"], row["patientid_his"], r_ckd['k_hyper'], row_dat['potassium_gp1'])
        elif len(row_ckd) == 0:
            row_dat["his_CKD_MDRD"], row_dat["his_CKD"] = -1, -1
            print("Empty CKD Mapping!", row["ReqNo"], row["patientid_his"])
        else:
            print("Multiple CKD Mapping!", row["ReqNo"], row["patientid_his"])
            print("Captured Rows:", row_ckd)

        # Add them to new DB
        dst_db_session.add(ECGtoK(**row_dat))
        dst_db_session.commit()
        move_lead_by_mhash(src_db_session, dst_db_session, row_dat["mhash"])


def add_lvh_data_1105(lvh_trainmul_data_loc: str,  # pylint: disable=too-many-statements
                      lvh_pretrain_data_loc: str,
                      lvh_testing_data_loc: str,
                      lvh_origsingle_data_loc: str,
                      lvh_origsingletest_data_loc: str,
                      src_db_session, dst_db_session):
    # Read LVH Data
    lvh_trainmul_data = pd.read_csv(lvh_trainmul_data_loc)
    lvh_pretrain_data = pd.read_csv(lvh_pretrain_data_loc)
    lvh_origsingle_data = pd.read_csv(lvh_origsingle_data_loc)

    # Get valid patients (Patients with single LVH label)
    lvh_his = lvh_trainmul_data.groupby("patientid_his")['LVH_level'].agg(
        [('LVH_history', lambda value: len(set(value)))]).reset_index()
    print("Total Golden Label Patients:", len(lvh_his))
    lvh_his = lvh_his[lvh_his["LVH_history"] == 1]
    print("Valid Golden Label Patients:", len(lvh_his))
    valid_patients = lvh_his["patientid_his"].to_list()
    lvh_trainmul_valid_data = lvh_trainmul_data[lvh_trainmul_data['patientid_his'].isin(valid_patients)]

    # Merge Golden Label and Pretrain Label data
    lvh_train_data = pd.merge(lvh_trainmul_valid_data, lvh_pretrain_data, how='outer',
                              on=['ReqNo', 'patientid_his', 'gender', 'EKG_age'])
    notna_cnt = pd.notna(lvh_train_data).sum()
    total_record_cnt = len(lvh_train_data)
    # Check Golden Label record data
    assert notna_cnt["LVmass_BSA"] == notna_cnt["LVH"] == notna_cnt["LVH_level"]
    assert notna_cnt["LVmass_BSA"] == len(lvh_trainmul_valid_data)
    print("Golden Label Records:", notna_cnt["LVmass_BSA"])
    # Check Pretrain Label record data
    assert notna_cnt["LVH_LVmass"] == notna_cnt["LVH_IVSd_PWT"]
    # This can catch errors if golden label record do not have pretrain label
    assert notna_cnt["LVH_LVmass"] == len(lvh_pretrain_data) == total_record_cnt
    print("Pretrain Label Records:", notna_cnt["LVH_LVmass"])

    # Join original single to label the original choice
    lvh_origsingle_valid_data = lvh_origsingle_data[lvh_origsingle_data['patientid_his'].isin(valid_patients)]
    # All these data shall already be in lvh_train, invalid patients shall have pretrain label but no golden label
    lvh_train_data = pd.merge(lvh_train_data, lvh_origsingle_valid_data, how='outer', on=['ReqNo', 'patientid_his'])
    assert len(lvh_train_data) == total_record_cnt

    # Fill na with -1 to indicate missing value
    lvh_train_data = lvh_train_data.fillna(-1)

    for _index, row in tqdm(lvh_train_data.iterrows(), total=len(lvh_train_data)):
        # Map LVH_Data to new Format
        row_dat = {
            "req_no": row["ReqNo"],
            "patient_id": row["patientid_his"],
        }
        row_dat["mhash"] = get_mhash(row_dat["req_no"], row_dat["patient_id"], "#")

        assert row["gender"] in ["M", "F"]
        row_dat["gender"] = 1 if row["gender"] == "M" else 0
        row_dat["EKG_age"] = int(row["EKG_age"])

        # Training Data set to 0
        row_dat["LVH_Testing"] = 0
        # If in original single record per patient set to 1 else 0
        row_dat["LVH_OriginalSingle"] = 0 if row["LVH_level_gp"] == -1 else 1

        # Unknown is set to -1 due to previously fillna with -1
        assert row["LVH"] in [-1, 0, 1]
        row_dat["LVH_LVmass_BSA"] = int(row["LVH"])
        assert row["LVH_level"] in [-1, 0, 1, 2, 3]
        row_dat["LVH_LVmass_BSA_level"] = int(row["LVH_level"])
        row_dat["LVH_LVmass_BSA_raw"] = float(row["LVmass_BSA"])
        assert row["LVH_LVmass"] in [-1, 0, 1]
        row_dat["LVH_LVmass"] = int(row["LVH_LVmass"])
        assert row["LVH_IVSd_PWT"] in [-1, 0, 1]
        row_dat["LVH_IVSd_PWT"] = int(row["LVH_IVSd_PWT"])

        # Add them to new DB
        dst_db_session.add(ECGtoLVH(**row_dat))
        dst_db_session.commit()
        move_lead_by_mhash(src_db_session, dst_db_session, row_dat["mhash"])

    # Read LVH testing Data
    lvh_testing_data = pd.read_csv(lvh_testing_data_loc)
    lvh_origsingletest_data = pd.read_csv(lvh_origsingletest_data_loc)
    lvh_origsingletest_data['orig_single'] = 1

    # Join original single to label the original choice
    total_record_cnt = len(lvh_testing_data)
    lvh_testing_data = pd.merge(lvh_testing_data, lvh_origsingletest_data, how='outer',
                                on=['ReqNo', 'patientid_his', 'gender', 'EKG_age'])
    assert len(lvh_testing_data) == total_record_cnt
    # Fill na with 0 to indicate not from single test
    lvh_testing_data['orig_single'] = lvh_testing_data['orig_single'].fillna(0)

    for _index, row in tqdm(lvh_testing_data.iterrows(), total=len(lvh_testing_data)):
        # Map LVH_Data to new Format
        row_dat = {
            "req_no": row["ReqNo"],
            "patient_id": row["patientid_his"],
        }
        row_dat["mhash"] = get_mhash(row_dat["req_no"], row_dat["patient_id"], "#")

        assert row["gender"] in ["M", "F"]
        row_dat["gender"] = 1 if row["gender"] == "M" else 0
        row_dat["EKG_age"] = int(row["EKG_age"])

        # Testing Data set to 1
        row_dat["LVH_Testing"] = 1
        # If in original single record per patient set to 1 else 0
        assert row["orig_single"] in [0, 1]
        row_dat["LVH_OriginalSingle"] = row["orig_single"]

        # Testing Data, Unknown is set to -1
        row_dat["LVH_LVmass_BSA"] = -1
        row_dat["LVH_LVmass_BSA_level"] = -1
        row_dat["LVH_LVmass_BSA_raw"] = -1
        row_dat["LVH_LVmass"] = -1
        row_dat["LVH_IVSd_PWT"] = -1

        # Add them to new DB
        dst_db_session.add(ECGtoLVH(**row_dat))
        dst_db_session.commit()
        move_lead_by_mhash(src_db_session, dst_db_session, row_dat["mhash"])


def add_lvh_data_full(lvh_full_data_loc: str, src_db_session, dst_db_session):  # pylint: disable=too-many-statements
    # Read LVH Data
    lvh_full_data = pd.read_csv(
        lvh_full_data_loc,
        parse_dates=["ECG_date", "echo_date", "echo_wt_date", "echo_ht_date", "ECG_wt_date", "ECG_ht_date"]
    )

    # Fill na with -1 to indicate missing value
    lvh_full_data = lvh_full_data.fillna(-1)

    # Find the first echo and nearest ECG to the echo for each patient
    # If same delta, choose the ECG after Echo; if multiple ECG after Echo order by ReqNo and pick first
    lvh_full_data["ECG_Echo_intertime"] = lvh_full_data["echo_date"] - lvh_full_data["ECG_date"]
    lvh_full_data["ECG_Echo_abs_intertime"] = abs(lvh_full_data["echo_date"] - lvh_full_data["ECG_date"])
    lvh_full_data = lvh_full_data.sort_values(
        by=["patientid_his", "echo_date", "ECG_Echo_abs_intertime", "ECG_Echo_intertime", "ReqNo"])
    lvh_full_data_pick = lvh_full_data[["patientid_his", "ReqNo"]].groupby('patientid_his').nth(0).reset_index()
    lvh_full_data_pick['picked'] = 1
    lvh_full_data = pd.merge(lvh_full_data, lvh_full_data_pick, how='outer', on=['patientid_his', 'ReqNo'])
    lvh_full_data['picked'] = lvh_full_data['picked'].fillna(0)
    lvh_full_data = lvh_full_data.reset_index()

    # DEBUG Usage
    # pd.set_option('display.max_rows', 200)
    # print(lvh_full_data[["patientid_his", "ReqNo", "echo_date", "ECG_date", "ECG_Echo_intertime", "picked"]].head(50))

    for _index, row in tqdm(lvh_full_data.iterrows(), total=len(lvh_full_data)):
        # Map LVH_Data to new Format
        row_dat = {
            "req_no": row["ReqNo"],
            "patient_id": row["patientid_his"],
        }
        row_dat["mhash"] = get_mhash(row_dat["req_no"], row_dat["patient_id"], "#")

        assert row["gender"] in ["M", "F"]
        row_dat["gender"] = 1 if row["gender"] == "M" else 0
        row_dat["EKG_age"] = int(row["EKG_age"])

        assert row["his_HTN_e"] in [0, 1]
        row_dat["his_HTN"] = row["his_HTN_e"]
        assert row["his_DM_e"] in [0, 1]
        row_dat["his_DM"] = row["his_DM_e"]
        assert row["his_MI_e"] in [0, 1]
        row_dat["his_MI"] = row["his_MI_e"]
        assert row["his_HF_e"] in [0, 1]
        row_dat["his_HF"] = row["his_HF_e"]
        assert row["his_stroke_e"] in [0, 1]
        row_dat["his_stroke"] = row["his_stroke_e"]
        assert row["his_CKD_e"] in [0, 1]
        row_dat["his_CKD"] = row["his_CKD_e"]

        # Training Data set to 0
        row_dat["LVH_Testing"] = 0

        # Unknown is set to -1 due to previously fillna with -1
        assert row["LVH_IVSd_PWT"] in [-1, 0, 1]
        row_dat["LVH_IVSd_PWT"] = int(row["LVH_IVSd_PWT"])

        row_dat["LVH_LVmass_raw"] = float(row["LV_mass"])
        if row["gender"] == "M":
            lvmass_bins = [225.0, 259.0, 293.0]
        else:
            lvmass_bins = [163.0, 187.0, 211.0]
        row_dat["LVH_LVmass_level"] = int(np.digitize(row_dat["LVH_LVmass_raw"], lvmass_bins))
        assert row_dat["LVH_LVmass_level"] in [0, 1, 2, 3]

        row_dat["Echo_Date"] = row["echo_date"]
        row_dat["ECG_Date"] = row["ECG_date"]

        def cal_BSA(height: float, weight: float):
            return math.sqrt((height * weight) / 3600)

        if row["gender"] == "M":
            lvmass_bsa_bins = [116.0, 132.0, 149.0]
        else:
            lvmass_bsa_bins = [96.0, 109.0, 122.0]

        if row["echoweight"] != -1 and row["echoheight"] != -1:
            assert row["echo_wt_date"] != -1
            row_dat["Weight_Echo_Date"] = row["echo_wt_date"]
            row_dat["Weight_Echo"] = float(row["echoweight"])
            assert row["echo_ht_date"] != -1
            row_dat["Height_Echo_Date"] = row["echo_ht_date"]
            row_dat["Height_Echo"] = float(row["echoheight"])
            row_dat["BSA_Echo"] = cal_BSA(row_dat["Height_Echo"], row_dat["Weight_Echo"])
            row_dat["LVH_LVmass_BSA_Echo_raw"] = row_dat["LVH_LVmass_raw"] / row_dat["BSA_Echo"]
            row_dat["LVH_LVmass_BSA_Echo_level"] = int(np.digitize(row_dat["LVH_LVmass_BSA_Echo_raw"], lvmass_bsa_bins))
            assert row_dat["LVH_LVmass_BSA_Echo_level"] in [0, 1, 2, 3]
        else:
            # row_dat["Weight_Echo_Date"], row_dat["Height_Echo_Date"]
            row_dat["Weight_Echo"] = -1
            row_dat["Height_Echo"] = -1
            row_dat["BSA_Echo"] = -1
            row_dat["LVH_LVmass_BSA_Echo_raw"] = -1
            row_dat["LVH_LVmass_BSA_Echo_level"] = -1

        if row["ECGweight"] != -1 and row["ECGheight"] != -1:
            assert row["ECG_wt_date"] != -1
            row_dat["Weight_ECG_Date"] = row["ECG_wt_date"]
            row_dat["Weight_ECG"] = float(row["ECGweight"])
            assert row["ECG_ht_date"] != -1
            row_dat["Height_ECG_Date"] = row["ECG_ht_date"]
            row_dat["Height_ECG"] = float(row["ECGheight"])
            row_dat["BSA_ECG"] = cal_BSA(row_dat["Height_ECG"], row_dat["Weight_ECG"])
            row_dat["LVH_LVmass_BSA_ECG_raw"] = row_dat["LVH_LVmass_raw"] / row_dat["BSA_ECG"]
            row_dat["LVH_LVmass_BSA_ECG_level"] = int(np.digitize(row_dat["LVH_LVmass_BSA_ECG_raw"], lvmass_bsa_bins))
            assert row_dat["LVH_LVmass_BSA_ECG_level"] in [0, 1, 2, 3]
        else:
            # row_dat["Weight_ECG_Date"], row_dat["Height_ECG_Date"]
            row_dat["Weight_ECG"] = -1
            row_dat["Height_ECG"] = -1
            row_dat["BSA_ECG"] = -1
            row_dat["LVH_LVmass_BSA_ECG_raw"] = -1
            row_dat["LVH_LVmass_BSA_ECG_level"] = -1

        # Pick ECG's BSA if available, else use Echo's BSA
        if row_dat["LVH_LVmass_BSA_ECG_raw"] != -1:
            row_dat["LVH_LVmass_BSA_raw"] = row_dat["LVH_LVmass_BSA_ECG_raw"]
            row_dat["LVH_LVmass_BSA_level"] = row_dat["LVH_LVmass_BSA_ECG_level"]
        else:
            row_dat["LVH_LVmass_BSA_raw"] = row_dat["LVH_LVmass_BSA_Echo_raw"]
            row_dat["LVH_LVmass_BSA_level"] = row_dat["LVH_LVmass_BSA_Echo_level"]

        # We choose one LVH label per patient
        assert row["picked"] in [0, 1]
        row_dat["LVH_Single"] = row["picked"]

        # Add them to new DB
        dst_db_session.add(ECGtoLVH(**row_dat))
        dst_db_session.commit()
        move_lead_by_mhash(src_db_session, dst_db_session, row_dat["mhash"])


def normalize(data):
    data = np.nan_to_num(data)  # removing NaNs and Infs
    data = data - np.mean(data)
    data = data / np.std(data)
    return data


def read_mitbih(filename,  # pylint: disable=too-many-branches, too-many-statements
                classes=('F', 'N', 'S', 'V', 'Q'),
                smote_flag=False,
                data_type="train",
                unsup=False,
                personal=True):
    # read data
    data = []
    samples = scipy.io.loadmat(filename + ".mat")
    if data_type in ["train", "val"]:
        samples = samples['s2s_mitbih_DS1']
    elif data_type == "test":
        samples = samples['s2s_mitbih_DS2']
    else:
        logger.warning("No matched data type!")

    values = samples[0]['seg_values']
    labels = samples[0]['seg_labels']

    if unsup and personal:  # Data for unsupervised contrastive learning
        data = []
        for splitted_hb in values:
            person_hbs = []
            for hb in splitted_hb:
                person_hbs.append(np.transpose(hb[0]))
            data.append(person_hbs)
        return data
    elif unsup is True and personal is False:  # Data for transformation-based unsupervised learning
        cnt = 0
        for splitted_hb in values:
            for hb in splitted_hb:
                if (np.max(np.array(hb[0], dtype='float32').transpose(1, 0), axis=-1) - np.min(
                        np.array(hb[0], dtype='float32').transpose(1, 0), axis=-1))[0] == 0:
                    cnt += 1
                else:
                    data.append(hb[0])
        logger.info("Number of wrong hb: %s", cnt)
        return data
    else:  # Data for supervised learning
        #  Add all segments(beats) together
        hb_cnt = 0
        for splitted_hb in values:
            for hb in splitted_hb:
                data.append(hb[0])
                hb_cnt = hb_cnt + 1
        logger.info("DS%s total heart beat count: %s", data_type, hb_cnt)

        #  Add all labels together
        label_cnt = 0
        t_lables = []
        for person_hb_labels in labels:
            for hb_label in person_hb_labels[0]:
                t_lables.append(str(hb_label))
                label_cnt = label_cnt + 1
        logger.info("DS%s total label count: %s", data_type, label_cnt)
        del values

        t_data = np.asarray(data, dtype='float32')
        shape_v = t_data.shape
        t_data = np.reshape(t_data, [shape_v[0], 1, -1])
        t_lables = np.array(t_lables)

        # only take specific classes
        data = np.asarray([], dtype='float32').reshape(0, 1, shape_v[1])
        labels = np.asarray([]).reshape(0)
        for cl in classes:
            tr_labels = np.where(t_lables == cl)[0]
            labels = np.concatenate((labels, t_lables[tr_labels]))
            data = np.concatenate((data, t_data[tr_labels]))

        if data_type in ["train", "val"]:
            data, X_test, labels, y_test = train_test_split(data, labels, test_size=0.1, random_state=666)
            if data_type == "val":
                return X_test, y_test

            if smote_flag:
                # over-sampling: SMOTE
                X_train = np.reshape(data, [data.shape[0] * data.shape[1], -1])
                y_train = labels

                nums = []
                for cl in classes:
                    nums.append(len(np.where(y_train == cl)[0]))

                # ratio={0:nums[0],1:nums[0],2:nums[0]}
                # ratio={0:7000,1:nums[1],2:7000,3:7000}
                ratio = {'N': nums[0], 'S': nums[0], 'V': nums[0]}
                sm = SMOTE(random_state=666, sampling_strategy=ratio, n_jobs=16)
                data, labels = sm.fit_sample(X_train, y_train)
                data = np.reshape(data, [data.shape[0], 1, -1])

        logger.info("Data shape: %s", data.shape)

        return data, labels


def read_ecg5000(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, _, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]


def extract_statments(x: str) -> Set[str]:
    # Load statement extraction mapping
    with open("./Data/ecg_mapping.json", "r") as f:
        statement_match_dict = json.load(f)
    extracted_statements = []
    for statement in unpack_statements(x):
        sid, sname, _exp = statement

        sname = sname.upper()
        sname = re.sub(r"[,.;]", " ", sname)
        sname = re.sub(r"(WITH)*\s*(HR|HEART RATE|RATE)[\s=:]*[0-9]+\s*(BPM|/MIN)*", " ", sname)
        sname = re.sub(r"(V-RATE|RATE|WITH)*\s*[0-9]+\s*(BPM|/MIN)", " ", sname)
        sname = re.sub(r"V-RATE[\s:]*[0-9]*\s*([-=][0-9]+)*", " ", sname)
        sname = re.sub(r"A-RATE[\s:]*[0-9]+(-[0-9]+)*", " ", sname)
        sname = re.sub(r"RATE[\s:]*[0-9]+(-[0-9]+)*", " ", sname)
        sname = re.sub(r"\(*PR\s*[>=]\s*[0-9]+(ms)*\)*", " ", sname)
        sname = re.sub(r"PR\s*[>=]\s*[0-9]+(ms)*\)*", " ", sname)
        sname = re.sub(r"PR INTERVAL\s*[0-9]+\s*(MSEC|MS|BPM|MESC)", " ", sname)
        sname = re.sub(r"NORMAL P AXIS", " ", sname)

        sname = re.sub(r"^BORDERLINE", " ", sname)
        sname = re.sub(r"^PROBABLE", " ", sname)
        sname = re.sub(r"^CONSIDER", " ", sname)
        sname = re.sub(r"^SUSPECT", " ", sname)

        sname = re.sub(r"\(\s*\)", " ", sname)
        sname = re.sub(r"\s+", " ", sname).strip()

        for did, match_filter in statement_match_dict.items():
            if sid in match_filter["c_id"] or sname in match_filter["c_name"] or sname in match_filter["c_id"]:
                # extracted_statements += [f"C#{d}" for d in did.split("^")]
                extracted_statements += did.split("^")
                continue

        for did, match_filter in statement_match_dict.items():
            for pf in match_filter["p_id"]:
                if sid.find(pf) != -1:
                    # extracted_statements += [f"PI#{d}" for d in did.split("^")]
                    extracted_statements += did.split("^")
                    continue
            for pf in match_filter["p_name"]:
                if sname.find(pf) != -1:
                    # extracted_statements += [f"PN#{d}" for d in did.split("^")]
                    extracted_statements += did.split("^")
                    continue

    return set(extracted_statements)

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ECGDL.datasets.ecg_data_model import create_db
# from ECGDL.datasets.ecg_data_model import add_k_ckd_data
# from ECGDL.datasets.ecg_data_model import add_lvh_data_1105
from ECGDL.datasets.ecg_data_model import add_lvh_data_full


if __name__ == "__main__":
    # Input: XML Parsing Database (Version 2019/05/19: 735819 Records)
    # MD5 Checksum: 7b9514106798cdf9d81b1cd33be7120b
    # Safe Data: 462382 + 206854 + 32789 + 12378 = 714403
    # >> select taint_code, taint_history, count(*) from ECGLeads group by taint_history order by taint_code;
    #   0 |                                           | 462382
    # 105 | STRANGE_XMLHASH                           | 206854
    # 205 | IGN_DUPLICATE_MHASH$cnt:1                 | 32789
    # 205 | IGN_DUPLICATE_MHASH$cnt:1#STRANGE_XMLHASH | 12378
    XML_DB_PATH = "/home/micro/ecg_data.db"
    XML_DB_LOC = f"sqlite:///{XML_DB_PATH}"

    # Output: ECG Database
    ECG_DB_PATH = "/tmp/ecg_lvh_data_20191206.db"
    ECG_DB_LOC = f"sqlite:///{ECG_DB_PATH}"

    # Check that we will not harm dataset
    assert not os.path.isfile(ECG_DB_PATH)
    # Create new DB
    create_db(ECG_DB_LOC, drop_tables=True)

    # Create session to xml database
    # TODO: Check MD5 Checksum before reading
    # TODO: Force DB to be Read Only to prevent accidental write
    xml_engine = create_engine(XML_DB_LOC)
    xml_Session = sessionmaker(bind=xml_engine)
    xml_session = xml_Session()

    # Create session to new database
    ecg_engine = create_engine(ECG_DB_LOC)
    ecg_Session = sessionmaker(bind=ecg_engine)
    ecg_session = ecg_Session()

    # Input: K Data csv (MD5 Checksum: 425d5a39cce8c6ff7f1be6167c4d80e5)
    K_DATA_LOC = "Data/raw_K/EKG_K_NCTU20190828.csv"
    # Input: CKD Data csv (MD5 Checksum: 086abf0562058bfc82bd633d8e03d7bd)
    CKD_DATA_LOC = "Data/raw_CKD/CKD_definition_NCTU20190828.csv"
    # Add K_CKD Data
    # add_k_ckd_data(K_DATA_LOC, CKD_DATA_LOC, xml_session, ecg_session)

    # Input: LVH Training Data csv (MD5 Checksum: 190fec606cfbb206c849aec44740a02b)
    LVH_TRAINMUL_DATA_LOC = "Data/raw_LVH/LVH_NCTU20191009_training_multiple.csv"
    # Input: LVH Pretrain Data csv (MD5 Checksum: db74d1ca776599e1396199e138b72f49)
    LVH_PRETRAIN_DATA_LOC = "Data/raw_LVH/LVH_NCTU20191105_pretrain.csv"
    # Input: LVH Testing Data csv (MD5 Checksum: 444a4616c4131bd007ab029434c10388)
    LVH_TESTING_DATA_LOC = "Data/raw_LVH/LVH_NCTU20191009_testing_multiple.csv"
    # Input: LVH Original Single Data csv (MD5 Checksum: ddddcaa4946c8d51f333657d1cc84d4a)
    LVH_TRAINSINGLE_DATA_LOC = "Data/raw_LVH/LVH_NCTU20191009_training_single.csv"
    # Input: LVH Original Single Testing Data csv (MD5 Checksum: 7f3718bc111aa385f53b392731024b31)
    LVH_TESTSINGLE_DATA_LOC = "Data/raw_LVH/LVH_NCTU20190906_testing.csv"
    # Add LVH Data
    # Shall use 1105 ECGtoLVH table definition
    # add_lvh_data_1105(LVH_TRAINMUL_DATA_LOC, LVH_PRETRAIN_DATA_LOC, LVH_TESTING_DATA_LOC,
    #                   LVH_TRAINSINGLE_DATA_LOC, LVH_TESTSINGLE_DATA_LOC, xml_session, ecg_session)

    # Input: LVH Full Data csv (MD5 Checksum: c108f93502778f77589d831861c131a1)
    LVH_FULL_DATA_LOC = "Data/raw_LVH/LVH_NCTU20191203_full.csv"
    # Add LVH Data
    add_lvh_data_full(LVH_FULL_DATA_LOC, xml_session, ecg_session)

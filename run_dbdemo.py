import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ECGDL.datasets.ecg_data_model import ECGtoK, ECGtoLVH, ECGLeads, TaintCode, unpack_leads


if __name__ == "__main__":
    # Setup Data Source (MD5 Checksum: a1af925cf5edca878912c0995d057a63)
    # db_path = "/home/micro/ecg_k-ckd-lvh_data_20190828.db"
    # Setup Data Source (MD5 Checksum: 0715ab809df1471a627ecc8edeaaee5a)
    db_path = "/home/micro/ecg_k-ckd-lvh_data_20191001.db"

    # Connect to Database
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    print("DB Connected!")

    # Get all possible mhashes that have LeadData and is safe to use
    q = session.query(ECGtoK.mhash).filter(ECGtoK.leads.has(ECGLeads.taint_code < TaintCode.DEFAULT_SAFE))
    mhash_data = pd.read_sql(q.statement, session.bind)
    print("Data Count:", len(mhash_data))
    print(mhash_data.head())

    # Query for Targeted mhash (Recommended)
    mhash = mhash_data["mhash"][0]
    item = session.query(ECGtoK).filter(ECGtoK.mhash == mhash).one()

    # Query for Targeted req_no and patient_id
    # item = session.query(ECGtoK).filter(ECGtoK.req_no==req_no).filter(ECGtoK.patient_id==patient_id).one()

    # Get metadata from query item
    # gender: Female -> 0, Male -> 1
    # pair_type: 'EKG before K' -> 1, 'K before EKG' -> 2, 'at the same time' -> 3
    print("Metadata:", item.gender, item.EKG_age, item.EKG_K_interhour, item.pair_type)

    # Get class from query item
    print("Class:", item.potassium_gp1, item.potassium_gp2, item.potassium_gp3, item.his_CKD_MDRD, item.his_CKD)

    # Get 12 Lead Data
    print(unpack_leads(item.leads.lead_data).keys())
    # print(unpack_leads(item.leads.lead_data))

    # Get all possible mhashes that have LeadData and is safe to use
    q = session.query(ECGtoLVH.mhash, ECGtoLVH.LVH).filter(
        ECGtoLVH.leads.has(ECGLeads.taint_code < TaintCode.DEFAULT_SAFE))
    mhash_data = pd.read_sql(q.statement, session.bind)
    target_cnt = mhash_data.groupby('LVH').count()["mhash"].reset_index()
    print("Data Count:", len(mhash_data))
    print("Target Count:\n", target_cnt)
    print(mhash_data.head())

    # Query for Targeted mhash (Recommended)
    mhash = mhash_data["mhash"][0]
    item = session.query(ECGtoLVH).filter(ECGtoLVH.mhash == mhash).one()

    # Query for Targeted req_no and patient_id
    # item = session.query(ECGtoLVH).filter(ECGtoLVH.req_no==req_no).filter(ECGtoLVH.patient_id==patient_id).one()

    # Get metadata from query item
    print("Metadata:", item.gender, item.EKG_age)

    # Get class from query item
    print("Class:", item.LVH)

    # Get 12 Lead Data
    print(unpack_leads(item.leads.lead_data).keys())
    # print(unpack_leads(item.leads.lead_data))

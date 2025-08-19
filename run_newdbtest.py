from pprint import pprint

import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ECGDL.datasets.ecg_data_model import ECGtoK, ECGLeads, TaintCode, unpack_leads


if __name__ == "__main__":
    # Database Location
    olddb_path = "/mnt/bigdata/201904_data/ecgk_data.db"
    newdb_path = "/mnt/bigdata/ecgk_data.db"

    # Connect to Database
    oldengine = create_engine(f"sqlite:///{olddb_path}")
    oldSession = sessionmaker(bind=oldengine)
    oldsession = oldSession()
    newengine = create_engine(f"sqlite:///{newdb_path}")
    newSession = sessionmaker(bind=newengine)
    newsession = newSession()

    # Check K Data is the same
    oldq = oldsession.query(ECGtoK)
    old_mhash_data = pd.read_sql(oldq.statement, oldsession.bind)
    print("Data Count:", len(old_mhash_data))
    newq = newsession.query(ECGtoK)
    new_mhash_data = pd.read_sql(newq.statement, newsession.bind)
    print("Data Count:", len(new_mhash_data))
    attrs = [
        "mhash", "req_no", "patient_id", "gender", "EKG_age", "EKG_K_interhour", "theshortest", "pair_type",
        "potassium_value", "potassium_gp1", "potassium_gp2", "potassium_gp3",
        "HR", "PR", "QRS", "QT", "QTc", "P", "T", "RR", "QRSd",
    ]
    new_mhash_data = new_mhash_data.sort_values(by=['mhash']).reset_index(drop=True)
    old_mhash_data = old_mhash_data.sort_values(by=['mhash']).reset_index(drop=True)
    print(old_mhash_data[attrs].equals(new_mhash_data[attrs]))

    # Get all possible mhashes that have LeadData
    oldq = oldsession.query(ECGtoK.mhash).filter(ECGtoK.leads.has())
    old_mhash_data = pd.read_sql(oldq.statement, oldsession.bind)["mhash"].tolist()
    print("Data Count:", len(old_mhash_data))

    # Get all possible mhashes that have LeadData
    newq = newsession.query(ECGtoK.mhash).filter(ECGtoK.leads.has(ECGLeads.taint_code < TaintCode.DEFAULT_SAFE))
    new_mhash_data = pd.read_sql(newq.statement, newsession.bind)["mhash"].tolist()
    print("Data Count:", len(new_mhash_data))

    mhash_data_diff = set(new_mhash_data) - set(old_mhash_data)
    pprint(mhash_data_diff)

    # Query for Targeted mhash (Recommended)
    mhash = "966daea1d909a933124f24e48e6868b5045b008ba663348672b5092b5adaf991"
    item = newsession.query(ECGtoK).filter(ECGtoK.mhash == mhash).one()

    # Get 12 Lead Data
    for k, v in unpack_leads(item.leads.lead_data).items():
        print(k, list(filter(lambda x: x != 0, v[:-500]))[:10])

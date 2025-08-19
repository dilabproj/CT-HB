from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from biosppy.signals import ecg

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ECGDL.datasets.ecg_data_model import ECGtoLVH, ECGLeads, TaintCode, unpack_leads, ECGtoK

LEAD_NAMES = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6", "aVF", "aVL", "aVR"]

def preprocess_leads(lead):
    # Preprocess lead data to remove baseline wander and high freq noise
    LEAD_SAMPLING_RATE = 500
    corrected_signal, _, _ = ecg.st.filter_signal(lead, 'butter', 'highpass', 2, 1, LEAD_SAMPLING_RATE)
    preprocessed_signal, _, _ = ecg.st.filter_signal(corrected_signal, 'butter', 'lowpass', 12, 35, LEAD_SAMPLING_RATE)
    return preprocessed_signal

def draw_all_lead(mhash, target_table, db_path, tlim=5000):
    # Get Lead data, this data must already be divided by 10 (Eg. SPxml)
    # Connect to Database
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    print("DB Connected!")

    # Get all possible mhashes that have LeadData and is safe to use
    # Select the targeted mhash
    item = session.query(target_table).filter(target_table.mhash == mhash).one()

    # Unpack Lead data and remove last LEAD_DROP_LEN
    dat = unpack_leads(item.leads.lead_data)
    fig, l_ax = plt.subplots(12, 1, figsize=(10*3, 2*3*12), sharex='col')
    for idx, lead_name in enumerate(LEAD_NAMES):
        ax = l_ax[idx]
        ax.title.set_text(f'Lead: {lead_name}')
        ax.set_xlim(0, tlim)
        ax.set_ylim(-50, 50)
        ax.plot(preprocess_leads(dat[lead_name][0:tlim]), linewidth=1)
        ax.set_aspect(100/10)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', color='#ed8080', linestyle='-')
        ax.grid(which='minor', color='#e1c8c8', linestyle=':')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.tight_layout()
    plt.savefig(f"./Image/{mhash}_12leads.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    db_path = "/home/micro/ecg_lvh_data_20200203.db"
    # Connect to Database
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    print("DB Connected!")

    # Get all possible mhashes that have LeadData and is safe to use
    q = session.query(ECGtoLVH.mhash).filter(ECGtoLVH.leads.has(
        ECGLeads.taint_code < TaintCode.DEFAULT_SAFE)).filter(ECGtoLVH.LVH_LVmass_level == 0)
    # Get all possible mhashes that have LeadData and is safe to use
    mhash_data = pd.read_sql(q.statement, session.bind)
    mhash = mhash_data["mhash"][0]
    draw_all_lead(mhash, ECGtoLVH, db_path)
    mhash = mhash_data["mhash"][1]
    draw_all_lead(mhash, ECGtoLVH, db_path)

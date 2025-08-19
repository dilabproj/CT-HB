from typing import List, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from ECGDL.const import LEAD_SAMPLING_RATE
from ECGDL.experiment import ExperimentConfig
from ECGDL.datasets.dataset import ECGDataset
from ECGDL.datasets.ecg_data_model import ECGtoLVH
from ECGDL.preprocess import preprocess_leads
from ECGDL.preprocess.transform import RandomSelectHB


if __name__ == "__main__":
    config = ExperimentConfig()

    config.db_path = "/home/micro/ecg_lvh_data_20200203.db"
    config.random_seed = 666

    # Settings for LVH classification
    config.target_table = ECGtoLVH
    config.stratify_attr = ('gender', 'EKG_age', 'his_HTN', 'his_DM', 'his_MI', 'his_HF', 'his_stroke', 'his_CKD')
    config.target_attr = 'LVH_LVmass_level'
    config.target_attr_transform = ECGDataset.transform_lvh_level
    config.train_test_patient_possible_overlap = False

    # Use Preprocessing Function
    config.preprocess_lead = preprocess_leads

    # Set hb_cnt to -1 to get all extracted heartbeats
    config.global_transform = (
        RandomSelectHB(hb_cnt=-1, sampling_rate=LEAD_SAMPLING_RATE),
    )

    ecgk_dataset = ECGDataset(
        db_location=config.db_path,
        target_table=config.target_table,
        target_attr=config.target_attr,
        target_attr_transform=config.target_attr_transform,
        stratify_attr=config.stratify_attr,
        preprocess_lead=config.preprocess_lead,
        transform=config.global_transform,
        name_mapping=config.target_name_map("short"),
        random_seed=config.random_seed)

    ecgk_loader = DataLoader(ecgk_dataset, batch_size=1)

    # Split 12-lead ECG into heart beat
    ecg_data = tqdm(ecgk_loader, dynamic_ncols=True, leave=False)
    dict_df: Dict[str, List[Any]] = {'mhash': [], 'drop': []}
    for data, target, mhash in ecg_data:
        dict_df['mhash'].append(mhash[0])
        dict_df['drop'].append(1)

    df = pd.DataFrame(dict_df)
    df.to_csv('./mhash_split_hb_lvmass_20200203.csv')

    # Plot heart beat distribution
    # bar_plt = df['hb'].value_counts().plot(kind="bar", subplots=True)
    # for idx, value in enumerate(df['hb'].value_counts()):
    #     ax.annotate(value, (idx, value), xytext=(0, 15), textcoords='offset points')
    # fig = bar_plt[0].get_figure()
    # fig.savefig('./countplot_1106.png')

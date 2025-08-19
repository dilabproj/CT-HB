import numpy as np

import torch

from sqlalchemy import create_engine
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker

from tqdm import tqdm

from ECGDL.const import LEAD_NAMES
from ECGDL.datasets.ecg_data_model import ECGtoK, unpack_leads, ECGLeads, TaintCode
from ECGDL.preprocess import preprocess_leads
from ECGDL.campp import GradCAM, GradCAMpp, visualize_cam_line, visualize_cam_scatter


if __name__ == '__main__':
    # Setup Model Parameters
    model_name = 'JamaModel'
    target_layer_name = 'residual_blocks'
    model_pickle_path = f'./models/{model_name}_ckpt_ep0020'
    img_saveroot = './Image'

    # Setup Testing objects Parameters

    # Define target class {1: hypo, 2: normal, 3: hyper}
    target_class = 1

    # Define target lead
    target_lead = "II"
    num_instances = 10

    # Setup Database Parameters
    db_path = "/home/micro/ecgk_data.db"

    # Setup Device Parameters
    gpu_device_id = "1"

    # Initialize Device
    device = torch.device(f"cuda:{gpu_device_id}")

    # Load model and map cuda to defined cuda
    model = torch.load(model_pickle_path, map_location=device)
    model.eval()

    # Initialize a model, model_dict and gradcam
    model_dict = {
        'type': model_name,
        'arch': model,
        'layer_name': target_layer_name,
        'input_size': (5000)
    }
    grad_cam = GradCAM(model_dict, device)
    gradcampp = GradCAMpp(model_dict, device)

    # Get one ECG record

    # Connect to Database
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Query number of instances by targeted class
    items = session.query(ECGtoK).filter(ECGtoK.leads.has(ECGLeads.taint_code < TaintCode.DEFAULT_SAFE))\
        .filter(ECGtoK.potassium_gp1 == target_class).order_by(func.random()).limit(num_instances).all()

    for item in tqdm(items):
        ecg_dict = unpack_leads(item.leads.lead_data)

        # Preprocess ecg
        ecg, raw_ecg = [], []
        for _, value in ecg_dict.items():
            ecg.append(preprocess_leads(value))
            raw_ecg.append(value)
        ecg = np.array(ecg, dtype='float32')[:, :5000]
        raw_ecg = np.array(raw_ecg, dtype='float32')[:, :5000]

        # Assign ecg to CUDA and unsqeeze for batch size
        torch_ecg = torch.from_numpy(ecg).unsqueeze(0).to(device)

        # GradCAM
        mask, logit = grad_cam(torch_ecg)
        predicted_class = logit.argmax(1)[-1]
        weights = mask.cpu().numpy().flatten()

        # File notes
        item_truth = f'gp1:{item.potassium_gp1}_gp2:{item.potassium_gp2}_gp3:{item.potassium_gp3}'
        file_notes = f'{model_name}_{item_truth}_predicted:{predicted_class}_{item.mhash}'

        # Scatter plot
        visualize_cam_scatter(
            weights[1000:2000], ecg[LEAD_NAMES.index(target_lead)][1000:2000], f'{img_saveroot}/GradCAM_{file_notes}')

        # Line plot
        visualize_cam_line(
            weights[1000:2000], ecg[LEAD_NAMES.index(target_lead)][1000:2000], f'{img_saveroot}/GradCAM_{file_notes}')

        # GradCAMpp
        mask_pp, logit = gradcampp(torch_ecg)
        predicted_class = logit.argmax(1)[-1]
        weights = mask_pp.cpu().numpy().flatten()

        # Scatter plot
        visualize_cam_scatter(
            weights[1000:2000], ecg[LEAD_NAMES.index(target_lead)][1000:2000], f'{img_saveroot}/GradCAMpp_{file_notes}')

        # Line plot
        visualize_cam_line(
            weights[1000:2500], ecg[LEAD_NAMES.index(target_lead)][1000:2500], f'{img_saveroot}/GradCAMpp_{file_notes}')

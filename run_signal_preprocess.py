from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ECGDL.const import LEAD_SAMPLING_RATE
from ECGDL.datasets.ecg_data_model import ECGtoK, unpack_leads
from ECGDL.preprocess import cut_signal, remove_baseline_wander, remove_highfreq_noise, get_oneheartbeat, demo_plot


if __name__ == '__main__':
    # Database Location
    db_path = "/mnt/bigdata/ecgk_data.db"

    # Connect to Database
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Get a LeadData
    mhash = "3a8140670e89a949b819306451bf6463fb5ad5021efa38de5f6c2c2b0ec32417"
    item = session.query(ECGtoK).filter(ECGtoK.mhash == mhash).one()

    # Lead data
    lead_data = unpack_leads(item.leads[0].lead_data)

    # Cut signal to length = 5000 ms
    cutted_signal = cut_signal(lead_data['II'], 5000)
    demo_plot(cutted_signal, "raw_signal")

    # Remove baseline wander (butterworth)
    corrected_signal = remove_baseline_wander(cutted_signal, 'butterworth_highpass')
    demo_plot(corrected_signal, "butterworth_highpass")

    # Remove baseline wander (median_filter1D)
    corrected_signal, baseline = remove_baseline_wander(cutted_signal, 'median_filter1D', return_baseline=True)
    demo_plot(baseline, "baseline")
    demo_plot(corrected_signal, "median_filter1D")

    # Remove high frequency noise (butterworth)
    corrected_signal = remove_highfreq_noise(cutted_signal, 'butterworth_lowpass')
    demo_plot(corrected_signal, "butterworth_lowpass")

    # Remove high frequency noise (FIR)
    corrected_signal = remove_highfreq_noise(cutted_signal, 'fir')
    demo_plot(corrected_signal, "fir")

    # Get one heart beat
    templates = get_oneheartbeat(corrected_signal, LEAD_SAMPLING_RATE)
    demo_plot(templates, "onehb")

    # All preprocess (low pass: butterworth)
    corrected_signal = remove_baseline_wander(cutted_signal, 'butterworth_highpass')
    corrected_signal = remove_highfreq_noise(corrected_signal, 'butterworth_lowpass')
    templates = get_oneheartbeat(corrected_signal, LEAD_SAMPLING_RATE)
    demo_plot(corrected_signal, "filtered_signal_butter")
    demo_plot(templates, "onehb_butter")

    # All preprocess (low pass: FIR)
    corrected_signal = remove_baseline_wander(cutted_signal, 'butterworth_highpass')
    corrected_signal = remove_highfreq_noise(corrected_signal, 'fir')
    templates = get_oneheartbeat(corrected_signal, LEAD_SAMPLING_RATE)
    demo_plot(corrected_signal, "filtered_signal_fir")
    demo_plot(templates, "onehb_fir")

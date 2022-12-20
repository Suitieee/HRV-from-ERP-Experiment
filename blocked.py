# Load NeuroKit and other useful packages
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
# import seaborn as sns
# %matplotlib inline
from neurokit2 import ecg_clean, eda_clean

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
plt.rcParams['font.size']= 13

subj = pd.read_csv('E://Jiang in 2022/审美ECG/清洗后/静息态/静息态subj.csv', header=None, encoding='gb2312')
filename = subj.iloc[:, 0]  # 将dataframe第一列转换成series
bio_info = {}
bio_out = pd.DataFrame({})
for i in filename:
    data = pd.read_csv('E://Jiang in 2022/审美ECG/清洗后/静息态/' + i + '.csv', header=None)
    bio_df = pd.DataFrame({})  # 初始化
    eda_signal_raw = data.iloc[:, 0]
    ecg_signal_raw = data.iloc[:, 1]

    # ecg_signal = ecg_clean(ecg_signal)
    eda_signal = eda_clean(eda_signal_raw, sampling_rate=200, method="neurokit")
    ecg_signal = ecg_clean(ecg_signal_raw, sampling_rate=200, method="pantompkins1985")

    # Process ecg
    ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=200)
    # plot = nk.ecg_plot(ecg_signals[:3000], sampling_rate=200)

    # Process rsp
    eda_signals, info = nk.eda_process(eda_signal, sampling_rate=200)
    # plot = nk.eda_plot(eda_signals[:], sampling_rate=200)

    ecg_info = nk.ecg_intervalrelated(ecg_signals)
    bio_df = pd.concat([bio_df, ecg_info],axis = 1)

    eda_info = nk.eda_intervalrelated(eda_signals)
    bio_df = pd.concat([bio_df, eda_info],axis = 1)

    bio_df.to_csv(i + "_output.csv")


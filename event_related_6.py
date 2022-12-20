# Load NeuroKit and other useful packages
import csv
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.io as scio  # 读取mat文件
import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
from neurokit2 import ecg_clean, ecg_process, events_create, ecg_analyze, ecg_eventrelated, epochs_to_df, eda_clean

plt.rcParams['figure.figsize'] = [8, 5]  # Bigger images
df_all = {}  # Initialize an empty dict,存储所有的index

subj = pd.read_csv('E://Jiang in 2022/审美ECG/清洗后/other/other端subj测试.csv', header=None)
# subj = pd.read_csv('E://Jiang in 2022/BIODATA/exp/subj.csv', header=None)
filename = subj.iloc[:,0] # 将dataframe第一列转换成series

for i in filename:
    # data = pd.read_csv('E://Jiang in 2022/审美ECG/清洗后/other/' + i + '.csv', header=None)
    data = pd.read_csv('E://Jiang in 2022/审美ECG/清洗后/other/' + i + '.csv', header=None)
    eda_signal_raw = data.iloc[:, 0]
    ecg_signal_raw = data.iloc[:, 1]
    event_channel_64 = data.iloc[:, 2]
    event_channel_32 = data.iloc[:, 3]
    event_channel_16 = data.iloc[:, 4]
    event_channel_8 = data.iloc[:, 5]
    event_channel_4 = data.iloc[:, 6]
    event_channel_2 = data.iloc[:, 7]

    # ecg_signal = ecg_clean(ecg_signal)
    eda_signal = eda_clean(eda_signal_raw, sampling_rate=200, method="neurokit")
    ecg_signal = ecg_clean(ecg_signal_raw, sampling_rate=200, method="pantompkins1985")


    # Find events
    events64 = nk.events_find(event_channel_64,threshold=4.9,threshold_keep="above",start_at=0,end_at=None,duration_min=700,
                            duration_max=900,inter_min=0,discard_first=0,discard_last=0,event_labels=None,event_conditions=None)
    events32 = nk.events_find(event_channel_32,threshold=4.9,threshold_keep="above",start_at=0,end_at=None,duration_min=700,
                           duration_max=900,inter_min=0,discard_first=0,discard_last=0,event_labels=None,event_conditions=None)
    events16 = nk.events_find(event_channel_16,threshold=4.9,threshold_keep="above",start_at=0,end_at=None,duration_min=700,
                          duration_max=900,inter_min=0,discard_first=0,discard_last=0,event_labels=None,event_conditions=None)
    events8  = nk.events_find(event_channel_8 ,threshold=4.9,threshold_keep="above",start_at=0,end_at=None,duration_min=700,
                           duration_max=900,inter_min=0,discard_first=0,discard_last=0,event_labels=None,event_conditions=None)
    events4  = nk.events_find(event_channel_4 ,threshold=4.9,threshold_keep="above",start_at=0,end_at=None,duration_min=700,
                           duration_max=900,inter_min=0,discard_first=0,discard_last=0,event_labels=None,event_conditions=None)
    events2  = nk.events_find(event_channel_2 ,threshold=4.9,threshold_keep="above",start_at=0,end_at=None,duration_min=700,
                           duration_max=900,inter_min=0,discard_first=0,discard_last=0,event_labels=None,event_conditions=None)

    # Process the signal
    df, info = nk.bio_process(ecg=ecg_signal, rsp=None, eda=eda_signal, sampling_rate=200)

    # Visualize
    # df.plot()
    # plot = nk.events_plot(df['SCR_Peaks'], ecg_signal)
    # plot = nk.events_plot(events64, ecg_signal)
    # plot = nk.events_plot(events32, ecg_signal)
    # plot = nk.events_plot(events16, ecg_signal)
    # plot = nk.events_plot(events8 , ecg_signal)
    # plot = nk.events_plot(events4 , ecg_signal)
    # plot = nk.events_plot(events2 , ecg_signal)

    # Epoch the data
    epochs64 = nk.epochs_create(df,events64,sampling_rate=200,epochs_start=-1,epochs_end=5)
    epochs32 = nk.epochs_create(df,events32,sampling_rate=200,epochs_start=-1,epochs_end=5)
    epochs16 = nk.epochs_create(df,events16,sampling_rate=200,epochs_start=-1,epochs_end=5)
    epochs8 = nk.epochs_create(df,events8 ,sampling_rate=200,epochs_start=-1,epochs_end=5)
    epochs4 = nk.epochs_create(df,events4 ,sampling_rate=200,epochs_start=-1,epochs_end=5)
    epochs2 = nk.epochs_create(df,events2 ,sampling_rate=200,epochs_start=-1,epochs_end=5)

    #绘制每一段epoch
    # for i, epoch in enumerate (epochs64):
       # epoch = epochs64[epoch]  # iterate epochs",
       # epoch = epoch[['ECG_Clean', 'ECG_Rate']]  # Select relevant columns",
       # title = events64['condition'][i] # get title from condition list",
       # nk.standardize(epoch).plot( legend=True)  # Plot scaled signals"
    # Visualize R-peaks in ECG signal
    # plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
    # Zooming into the first 5 R-peaks
    # plot = nk.events_plot(rpeaks['ECG_R_Peaks'][:5], ecg_signal[:20000])

    df64 = {}  # Initialize an empty dict,
    scr_num_64_4s = 0
    scr_num_64_5s = 0
    scr_num_diff_64_4s = 0
    scr_num_diff_64_5s = 0
    for epoch_index in epochs64:
        df64[epoch_index] = {}  # then Initialize an empty dict inside of it with the iterative

        # Save a temp var with dictionary called <epoch_index> in epochs-dictionary
        epoch = epochs64[epoch_index]

        # We want its features:

        # Feature 1 ECG
        # hrv_time = nk.hrv_time(epoch["ECG_R_Peaks"], sampling_rate=200, show=True)

        ecg_baseline = epoch["ECG_Rate"].loc[-200:0].mean()  # Baseline
        ecg_mean = epoch["ECG_Rate"].loc[0:800].mean()  # Mean heart rate in the 0-4 seconds
        # Store ECG in df
        df64[epoch_index]["ECG_Rate"] = ecg_mean   # Correct for baseline
        df64[epoch_index]["ECG_Rate_minus"] = ecg_mean - ecg_baseline   # Correct for baseline

        # Feature 2 EDA - SCR
        scr_max_4s = epoch["SCR_Amplitude"].loc[0:800].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_4s):
            scr_max_4s = 0
        else:
            scr_num_64_4s = scr_num_64_4s + 1
        # Store SCR in df
        df64[epoch_index]["SCR_Magnitude_4s"] = scr_max_4s

        scr_max_5s = epoch["SCR_Amplitude"].loc[0:1000].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_5s):
            scr_max_5s = 0
        else:
            scr_num_64_5s = scr_num_64_5s + 1
        # Store SCR in df
        df64[epoch_index]["SCR_Magnitude_5s"] = scr_max_5s

        df64[epoch_index]["SCR_diff_4s"] = epoch['EDA_Clean'].loc[0:800].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df64[epoch_index]["SCR_diff_4s"] < 0.01:
            df64[epoch_index]["SCR_diff_4s"] = 0
        else:
            df64[epoch_index]["SCR_diff_4s"] = df64[epoch_index]["SCR_diff_4s"] ** 0.5
            scr_num_diff_64_4s = scr_num_diff_64_4s + 1

        df64[epoch_index]["SCR_diff_5s"] = epoch['EDA_Clean'].loc[0:1000].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df64[epoch_index]["SCR_diff_5s"] < 0.01:
            df64[epoch_index]["SCR_diff_5s"] = 0
        else:
            df64[epoch_index]["SCR_diff_5s"] = df64[epoch_index]["SCR_diff_5s"] ** 0.5
            scr_num_diff_64_5s = scr_num_diff_64_5s + 1
    df64 = pd.DataFrame.from_dict(df64, orient="index")  # Convert to a dataframe
    df64  # Print DataFrame


    df32 = {}   # Initialize an empty dict,
    scr_num_32_4s = 0
    scr_num_32_5s = 0
    scr_num_diff_32_4s = 0
    scr_num_diff_32_5s = 0
    for epoch_index in epochs32:
        df32[epoch_index] = {}  # then Initialize an empty dict inside of it with the iterative

        # Save a temp var with dictionary called <epoch_index> in epochs-dictionary
        epoch = epochs32[epoch_index]

            # We want its features:

        # Feature 1 ECG
        ecg_baseline = epoch["ECG_Rate"].loc[-200:0].mean()  # Baseline
        ecg_mean = epoch["ECG_Rate"].loc[0:800].mean()  # Mean heart rate in the 0-4 seconds
        # Store ECG in df
        df32[epoch_index]["ECG_Rate"] = ecg_mean   # Correct for baseline
        df32[epoch_index]["ECG_Rate_minus"] = ecg_mean - ecg_baseline   # Correct for baseline

        # Feature 2 EDA - SCR
        scr_max_4s = epoch["SCR_Amplitude"].loc[0:800].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_4s):
            scr_max_4s = 0
        else:
            scr_num_32_4s = scr_num_32_4s + 1
        # Store SCR in df
        df32[epoch_index]["SCR_Magnitude_4s"] = scr_max_4s

        scr_max_5s = epoch["SCR_Amplitude"].loc[0:1000].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_5s):
            scr_max_5s = 0
        else:
            scr_num_32_5s = scr_num_32_5s + 1
        # Store SCR in df
        df32[epoch_index]["SCR_Magnitude_5s"] = scr_max_5s

        df32[epoch_index]["SCR_diff_4s"] = epoch['EDA_Clean'].loc[0:800].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df32[epoch_index]["SCR_diff_4s"] < 0.01:
            df32[epoch_index]["SCR_diff_4s"] = 0
        else:
            df32[epoch_index]["SCR_diff_4s"] = df32[epoch_index]["SCR_diff_4s"] ** 0.5
            scr_num_diff_32_4s = scr_num_diff_32_4s + 1

        df32[epoch_index]["SCR_diff_5s"] = epoch['EDA_Clean'].loc[0:1000].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df32[epoch_index]["SCR_diff_5s"] < 0.01:
            df32[epoch_index]["SCR_diff_5s"] = 0
        else:
            df32[epoch_index]["SCR_diff_5s"] = df32[epoch_index]["SCR_diff_5s"] ** 0.5
            scr_num_diff_32_5s = scr_num_diff_32_5s + 1
    df32 = pd.DataFrame.from_dict(df32, orient="index")  # Convert to a dataframe
    df32  # Print DataFrame

    df16 = {}  # Initialize an empty dict,
    scr_num_16_4s = 0
    scr_num_16_5s = 0
    scr_num_diff_16_4s = 0
    scr_num_diff_16_5s = 0
    for epoch_index in epochs16:
        df16[epoch_index] = {}  # then Initialize an empty dict inside of it with the iterative

        # Save a temp var with dictionary called <epoch_index> in epochs-dictionary
        epoch = epochs16[epoch_index]

       # We want its features:

       # Feature 1 ECG
        ecg_baseline = epoch["ECG_Rate"].loc[-200:0].mean()  # Baseline
        ecg_mean = epoch["ECG_Rate"].loc[0:800].mean()  # Mean heart rate in the 0-4 seconds
        # Store ECG in df
        df16[epoch_index]["ECG_Rate"] = ecg_mean   # Correct for baseline
        df16[epoch_index]["ECG_Rate_minus"] = ecg_mean - ecg_baseline   # Correct for baseline

        # Feature 2 EDA - SCR
        scr_max_4s = epoch["SCR_Amplitude"].loc[0:800].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_4s):
            scr_max_4s = 0
        else:
            scr_num_16_4s = scr_num_16_4s + 1
        # Store SCR in df
        df16[epoch_index]["SCR_Magnitude_4s"] = scr_max_4s

        scr_max_5s = epoch["SCR_Amplitude"].loc[0:1000].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_5s):
            scr_max_5s = 0
        else:
            scr_num_16_5s = scr_num_16_5s + 1
        # Store SCR in df
        df16[epoch_index]["SCR_Magnitude_5s"] = scr_max_5s

        df16[epoch_index]["SCR_diff_4s"] = epoch['EDA_Clean'].loc[0:800].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df16[epoch_index]["SCR_diff_4s"] < 0.01:
            df16[epoch_index]["SCR_diff_4s"] = 0
        else:
            df16[epoch_index]["SCR_diff_4s"] = df16[epoch_index]["SCR_diff_4s"] ** 0.5
            scr_num_diff_16_4s = scr_num_diff_16_4s + 1

        df16[epoch_index]["SCR_diff_5s"] = epoch['EDA_Clean'].loc[0:1000].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df16[epoch_index]["SCR_diff_5s"] < 0.01:
            df16[epoch_index]["SCR_diff_5s"] = 0
        else:
            df16[epoch_index]["SCR_diff_5s"] = df16[epoch_index]["SCR_diff_5s"] ** 0.5
            scr_num_diff_16_5s = scr_num_diff_16_5s + 1
    df16 = pd.DataFrame.from_dict(df16, orient="index")  # Convert to a dataframe
    df16  # Print DataFrame

    df8 = {}  # Initialize an empty dict,
    scr_num_8_4s = 0
    scr_num_8_5s = 0
    scr_num_diff_8_4s = 0
    scr_num_diff_8_5s = 0
    for epoch_index in epochs8:
        df8[epoch_index] = {}  # then Initialize an empty dict inside of it with the iterative

        # Save a temp var with dictionary called <epoch_index> in epochs-dictionary
        epoch = epochs8[epoch_index]

        # We want its features:

        # Feature 1 ECG
        ecg_baseline = epoch["ECG_Rate"].loc[-200:0].mean()  # Baseline
        ecg_mean = epoch["ECG_Rate"].loc[0:800].mean()  # Mean heart rate in the 0-4 seconds
        # Store ECG in df
        df8[epoch_index]["ECG_Rate"] = ecg_mean   # Correct for baseline
        df8[epoch_index]["ECG_Rate_minus"] = ecg_mean - ecg_baseline   # Correct for baseline

        # Feature 2 EDA - SCR
        scr_max_4s = epoch["SCR_Amplitude"].loc[0:800].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_4s):
            scr_max_4s = 0
        else:
            scr_num_8_4s = scr_num_8_4s + 1
        # Store SCR in df
        df8[epoch_index]["SCR_Magnitude_4s"] = scr_max_4s

        scr_max_5s = epoch["SCR_Amplitude"].loc[0:1000].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_5s):
            scr_max_5s = 0
        else:
            scr_num_8_5s = scr_num_8_5s + 1
        # Store SCR in df
        df8[epoch_index]["SCR_Magnitude_5s"] = scr_max_5s

        df8[epoch_index]["SCR_diff_4s"] = epoch['EDA_Clean'].loc[0:800].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df8[epoch_index]["SCR_diff_4s"] < 0.01:
            df8[epoch_index]["SCR_diff_4s"] = 0
        else:
            df8[epoch_index]["SCR_diff_4s"] = df8[epoch_index]["SCR_diff_4s"] ** 0.5
            scr_num_diff_8_4s = scr_num_diff_8_4s + 1

        df8[epoch_index]["SCR_diff_5s"] = epoch['EDA_Clean'].loc[0:1000].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df8[epoch_index]["SCR_diff_5s"] < 0.01:
            df8[epoch_index]["SCR_diff_5s"] = 0
        else:
            df8[epoch_index]["SCR_diff_5s"] = df8[epoch_index]["SCR_diff_5s"] ** 0.5
            scr_num_diff_8_5s = scr_num_diff_8_5s + 1
    df8 = pd.DataFrame.from_dict(df8, orient="index")  # Convert to a dataframe
    df8  # Print DataFrame

    df4 = {}  # Initialize an empty dict,
    scr_num_4_4s = 0
    scr_num_4_5s = 0
    scr_num_diff_4_4s = 0
    scr_num_diff_4_5s = 0
    for epoch_index in epochs4:
        df4[epoch_index] = {}  # then Initialize an empty dict inside of it with the iterative

        # Save a temp var with dictionary called <epoch_index> in epochs-dictionary
        epoch = epochs4[epoch_index]

        # We want its features:

        # Feature 1 ECG
        ecg_baseline = epoch["ECG_Rate"].loc[-200:0].mean()  # Baseline
        ecg_mean = epoch["ECG_Rate"].loc[0:800].mean()  # Mean heart rate in the 0-4 seconds
        # Store ECG in df
        df4[epoch_index]["ECG_Rate"] = ecg_mean   # Correct for baseline
        df4[epoch_index]["ECG_Rate_minus"] = ecg_mean - ecg_baseline   # Correct for baseline

        # Feature 2 EDA - SCR
        scr_max_4s = epoch["SCR_Amplitude"].loc[0:800].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_4s):
            scr_max_4s = 0
        else:
            scr_num_4_4s = scr_num_4_4s + 1
        # Store SCR in df
        df4[epoch_index]["SCR_Magnitude_4s"] = scr_max_4s

        scr_max_5s = epoch["SCR_Amplitude"].loc[0:1000].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_5s):
            scr_max_5s = 0
        else:
            scr_num_4_5s = scr_num_4_5s + 1
        # Store SCR in df
        df4[epoch_index]["SCR_Magnitude_5s"] = scr_max_5s

        df4[epoch_index]["SCR_diff_4s"] = epoch['EDA_Clean'].loc[0:800].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df4[epoch_index]["SCR_diff_4s"] < 0.01:
            df4[epoch_index]["SCR_diff_4s"] = 0
        else:
            df4[epoch_index]["SCR_diff_4s"] = df4[epoch_index]["SCR_diff_4s"] ** 0.5
            scr_num_diff_4_4s = scr_num_diff_4_4s + 1

        df4[epoch_index]["SCR_diff_5s"] = epoch['EDA_Clean'].loc[0:1000].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df4[epoch_index]["SCR_diff_5s"] < 0.01:
            df4[epoch_index]["SCR_diff_5s"] = 0
        else:
            df4[epoch_index]["SCR_diff_5s"] = df4[epoch_index]["SCR_diff_5s"] ** 0.5
            scr_num_diff_4_5s = scr_num_diff_4_5s + 1
    df4 = pd.DataFrame.from_dict(df4, orient="index")  # Convert to a dataframe
    df4  # Print DataFrame

    df2 = {}  # Initialize an empty dict,
    scr_num_2_4s = 0
    scr_num_2_5s = 0
    scr_num_diff_2_4s = 0
    scr_num_diff_2_5s = 0
    for epoch_index in epochs2:
        df2[epoch_index] = {}  # then Initialize an empty dict inside of it with the iterative

        # Save a temp var with dictionary called <epoch_index> in epochs-dictionary
        epoch = epochs2[epoch_index]

        # We want its features:

        # Feature 1 ECG
        ecg_baseline = epoch["ECG_Rate"].loc[-200:0].mean()  # Baseline
        ecg_mean = epoch["ECG_Rate"].loc[0:800].mean()  # Mean heart rate in the 0-4 seconds
        # Store ECG in df
        df2[epoch_index]["ECG_Rate"] = ecg_mean   # Correct for baseline
        df2[epoch_index]["ECG_Rate_minus"] = ecg_mean - ecg_baseline   # Correct for baseline

        # Feature 2 EDA - SCR
        scr_max_4s = epoch["SCR_Amplitude"].loc[0:800].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_4s):
            scr_max_4s = 0
        else:
            scr_num_2_4s = scr_num_2_4s + 1
        # Store SCR in df
        df2[epoch_index]["SCR_Magnitude_4s"] = scr_max_4s

        scr_max_5s = epoch["SCR_Amplitude"].loc[0:1000].max()  # Maximum SCR peak
        # If no SCR, consider the magnitude, i.e.  that the value is 0
        if np.isnan(scr_max_5s):
            scr_max_5s = 0
        else:
            scr_num_2_5s = scr_num_2_5s + 1
        # Store SCR in df
        df2[epoch_index]["SCR_Magnitude_5s"] = scr_max_5s

        df2[epoch_index]["SCR_diff_4s"] = epoch['EDA_Clean'].loc[0:800].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df2[epoch_index]["SCR_diff_4s"] < 0.01:
            df2[epoch_index]["SCR_diff_4s"] = 0
        else:
            df2[epoch_index]["SCR_diff_4s"] = df2[epoch_index]["SCR_diff_4s"] ** 0.5
            scr_num_diff_2_4s = scr_num_diff_2_4s + 1

        df2[epoch_index]["SCR_diff_5s"] = epoch['EDA_Clean'].loc[0:1000].max()-epoch["EDA_Clean"].loc[-200:0].mean()
        if df2[epoch_index]["SCR_diff_5s"] < 0.01:
            df2[epoch_index]["SCR_diff_5s"] = 0
        else:
            df2[epoch_index]["SCR_diff_5s"] = df2[epoch_index]["SCR_diff_5s"] ** 0.5
            scr_num_diff_2_5s = scr_num_diff_2_5s + 1
    df2 = pd.DataFrame.from_dict(df2, orient="index")  # Convert to a dataframe
    df2  # Print DataFrame

    ECG_Rate_mean_64 = df64["ECG_Rate"].mean()
    ECG_Rate_mean_32 = df32["ECG_Rate"].mean()
    ECG_Rate_mean_16 = df16["ECG_Rate"].mean()
    ECG_Rate_mean_8 = df8["ECG_Rate"].mean()
    ECG_Rate_mean_4 = df4["ECG_Rate"].mean()
    ECG_Rate_mean_2 = df2["ECG_Rate"].mean()

    ECG_Rate_minus_mean_64 = df64["ECG_Rate_minus"].mean()
    ECG_Rate_minus_mean_32 = df32["ECG_Rate_minus"].mean()
    ECG_Rate_minus_mean_16 = df16["ECG_Rate_minus"].mean()
    ECG_Rate_minus_mean_8 = df8["ECG_Rate_minus"].mean()
    ECG_Rate_minus_mean_4 = df4["ECG_Rate_minus"].mean()
    ECG_Rate_minus_mean_2 = df2["ECG_Rate_minus"].mean()

    EDA_SCR_Magnitude_mean_64_4s = df64["SCR_Magnitude_4s"].sum()/scr_num_64_4s
    EDA_SCR_Magnitude_mean_32_4s = df32["SCR_Magnitude_4s"].sum()/scr_num_32_4s
    EDA_SCR_Magnitude_mean_16_4s = df16["SCR_Magnitude_4s"].sum()/scr_num_16_4s
    EDA_SCR_Magnitude_mean_8_4s = df8["SCR_Magnitude_4s"].sum()/scr_num_8_4s
    EDA_SCR_Magnitude_mean_4_4s = df4["SCR_Magnitude_4s"].sum()/scr_num_4_4s
    EDA_SCR_Magnitude_mean_2_4s = df2["SCR_Magnitude_4s"].sum()/scr_num_2_4s

    EDA_SCR_Magnitude_mean_64_5s = df64["SCR_Magnitude_5s"].sum()/scr_num_64_5s
    EDA_SCR_Magnitude_mean_32_5s = df32["SCR_Magnitude_5s"].sum()/scr_num_32_5s
    EDA_SCR_Magnitude_mean_16_5s = df16["SCR_Magnitude_5s"].sum()/scr_num_16_5s
    EDA_SCR_Magnitude_mean_8_5s = df8["SCR_Magnitude_5s"].sum()/scr_num_8_5s
    EDA_SCR_Magnitude_mean_4_5s = df4["SCR_Magnitude_5s"].sum()/scr_num_4_5s
    EDA_SCR_Magnitude_mean_2_5s = df2["SCR_Magnitude_5s"].sum()/scr_num_2_5s

    EDA_SCR_diff_64_4s = df64["SCR_diff_4s"].sum() / scr_num_diff_64_4s
    EDA_SCR_diff_32_4s = df64["SCR_diff_4s"].sum() / scr_num_diff_32_4s
    EDA_SCR_diff_16_4s = df64["SCR_diff_4s"].sum() / scr_num_diff_16_4s
    EDA_SCR_diff_8_4s = df64["SCR_diff_4s"].sum() / scr_num_diff_8_4s
    EDA_SCR_diff_4_4s = df64["SCR_diff_4s"].sum() / scr_num_diff_4_4s
    EDA_SCR_diff_2_4s = df64["SCR_diff_4s"].sum() / scr_num_diff_2_4s

    EDA_SCR_diff_64_5s = df64["SCR_diff_5s"].sum() / scr_num_diff_64_5s
    EDA_SCR_diff_32_5s = df64["SCR_diff_5s"].sum() / scr_num_diff_32_5s
    EDA_SCR_diff_16_5s = df64["SCR_diff_5s"].sum() / scr_num_diff_16_5s
    EDA_SCR_diff_8_5s = df64["SCR_diff_5s"].sum() / scr_num_diff_8_5s
    EDA_SCR_diff_4_5s = df64["SCR_diff_5s"].sum() / scr_num_diff_4_5s
    EDA_SCR_diff_2_5s = df64["SCR_diff_5s"].sum() / scr_num_diff_2_5s

    trial_64 = len(df64["ECG_Rate"])
    trial_32 = len(df32["ECG_Rate"])
    trial_16 = len(df16["ECG_Rate"])
    trial_8 = len(df8["ECG_Rate"])
    trial_4 = len(df4["ECG_Rate"])
    trial_2 = len(df2["ECG_Rate"])

    # 输出目标
    # trial代表试次数量，scr_num代表拥有有效SCR的trial数量，ECG_Rate_mean代表4s刺激呈现内的平均实时心率
    # ECG_Rate_minus_mean代表4s实时平均心率减去pre1s实时平均心率，EDA_SCR_Magnitude_Mean代表内平均SCR【平均存疑】
    df_all[i] = (trial_64, scr_num_64_4s, scr_num_64_5s, ECG_Rate_mean_64, ECG_Rate_minus_mean_64,
                 EDA_SCR_Magnitude_mean_64_4s, EDA_SCR_Magnitude_mean_64_5s, EDA_SCR_diff_64_4s, EDA_SCR_diff_64_5s,

                 trial_32, scr_num_32_4s, scr_num_32_5s, ECG_Rate_mean_32, ECG_Rate_minus_mean_32,
                 EDA_SCR_Magnitude_mean_32_4s, EDA_SCR_Magnitude_mean_32_5s, EDA_SCR_diff_32_4s, EDA_SCR_diff_32_5s,

                 trial_16, scr_num_16_4s, scr_num_16_5s, ECG_Rate_mean_16, ECG_Rate_minus_mean_16,
                 EDA_SCR_Magnitude_mean_16_4s, EDA_SCR_Magnitude_mean_16_5s, EDA_SCR_diff_16_4s, EDA_SCR_diff_16_5s,

                 trial_8, scr_num_8_4s, scr_num_8_5s, ECG_Rate_mean_8, ECG_Rate_minus_mean_8,
                 EDA_SCR_Magnitude_mean_8_4s, EDA_SCR_Magnitude_mean_8_5s, EDA_SCR_diff_8_4s, EDA_SCR_diff_8_5s,

                 trial_4, scr_num_4_4s, scr_num_4_5s, ECG_Rate_mean_4, ECG_Rate_minus_mean_4,
                 EDA_SCR_Magnitude_mean_4_4s, EDA_SCR_Magnitude_mean_4_5s, EDA_SCR_diff_4_4s, EDA_SCR_diff_4_5s,

                 trial_2, scr_num_2_4s, scr_num_2_5s, ECG_Rate_mean_2, ECG_Rate_minus_mean_2,
                 EDA_SCR_Magnitude_mean_2_4s, EDA_SCR_Magnitude_mean_2_5s, EDA_SCR_diff_2_4s, EDA_SCR_diff_2_5s)

df_all_output = pd.DataFrame.from_dict(df_all, orient="index")  # Convert to a dataframe
df_all_output.to_csv("phone_output_9.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob, os
from poser.patients import Patient, PatientCollection
import re

import scipy.stats as st

import seaborn as sns



#%% load patient data

directory = './data/' # put the DLC tracking files in this directory


# this is the extension you want to detect
extension = '.csv'
files = []
for root, dirs_list, files_list in os.walk(directory):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            files.append(os.path.join(root, file_name))

fs = 160

patients = []
for file in files:
    
    split_name = file.split('/')[-1].split('_0deg')[0].split('_')
    split_name[:] = [x for x in split_name if x]
    
    
    speed = [s for s in split_name if 'cm' in s][0]
    speed =re.findall(r'\d+', speed)[0]

    sample_id = split_name[1]


    if 'OFF' in '\t'.join(split_name):
        stimpower=0
        stimcond = 'stimOFF'
    elif 'Baseline' in '\t'.join(split_name):
        stimpower = 0
        stimcond = 'baseline'
    else: 
        stimpower = [s for s in split_name if 'uA' in s][0]
        stimpower =re.findall(r'\d+', stimpower)[0]
        stimcond = 'stimON'
        
    cohort = 1    
    virus = split_name[0]
    
    patient_id = sample_id + '-cohort-' + str(cohort)  + '-cond-' + stimcond + '-' + str(stimpower) + 'uA-' + speed + 'cms-' + virus

    

    print(patient_id)
    
    #print("loading: Patient ID: {}".format(patient_id))
    pose_estimation = pd.read_csv( file ,header=[1,2])
    
    label = {'stim': stimcond,
             'stimpower':stimpower,
             'speed':int(speed),
             'cohort':cohort,
             'virus': virus}
    
    
    lat_columns = []
    ventral_columns = []            
    for col in pose_estimation.columns:
        if 'lat' in col[0]:
            lat_columns.append(col[0])
        elif 'ventral' in col[0]:
            ventral_columns.append(col[0])  
    
            

    markers = list(set(pose_estimation.columns.droplevel(1).tolist()))
    markers_ventral = [u for u in markers if 'ventral' in u]
    markers_lateral = [u for u in markers if 'lateral' in u]




    if pose_estimation.shape[0]>100:
        
        pose_estimation = pd.read_csv( file ,header=[1,2])

        
        
        p = Patient(pose_estimation,
                        fs,
                        patient_id=patient_id,
                        label=label,
                        label_name=None,                                
                        low_cut=0,
                        high_cut=None,
                        likelihood_cutoff=0.25,
                        normalize=False,
                        scaling_factor=1,                                    
                        interpolate_pose=True,
                        spike_threshold=0,
                        )
        
        
        pose_estimation = pd.read_csv( file ,header=[1,2])

        window = 5
        for marker in [*markers_ventral, *markers_lateral]:
            pose_estimation[(marker,'likelihood')] = pose_estimation[(marker,'likelihood')].rolling(window, min_periods=(window//2), center=True).mean() 

        
        p_test = Patient(pose_estimation,
                        fs,
                        patient_id=patient_id,
                        label=label,
                        label_name=None,                                
                        low_cut=0,
                        high_cut=None,
                        likelihood_cutoff=0.25,
                        normalize=False,
                        scaling_factor=1,                                    
                        interpolate_pose=False,
                        spike_threshold=0,
                        )
        
        p.raw_pose_estimation = p_test.pose_estimation
        
        patients.append(p)





#%% collect all patients into a collection object

pc = PatientCollection()
pc.add_patient_list(patients)
print("We have {} patients".format(len(pc)))




#%% define structural features to compute

from poser.structural_features import StructuralFeatures

sf = StructuralFeatures(markers = pc.markers)
sf.load_structural_features('structural_feature_files',folder = './')

#%% extract features

from poser.extraction import extract
pc = extract(pc, sf, normalize_features=False)


#%% aggregate features


from poser.feature_aggregation import aggregate
features_df= aggregate(pc)


#%%

def ensure_peak_to_trough(peaks,troughs):
    
    if peaks[0] < troughs[0]:
        reverse = False
        x = peaks.tolist()
        y = troughs.tolist()
    else:
        reverse = True
        x = troughs.tolist()
        y = peaks.tolist()
        
    for r in range(100):
        for i in range(len(x)-1):
            try:          
                if x[i+1] < y[i]:
                    x.pop(i+1)
            except:
                continue
            
        for i in range(len(y)-1):
            try:          
                if y[i+1] < x[i+1]:
                    y.pop(i+1)
            except:
                continue  
            
    min_idx = np.min([len(x),len(y)])
    x = x[:min_idx]
    y = y[:min_idx]

    if reverse:
        return np.array(y), np.array(x)
    else:
        return np.array(x), np.array(y)
    
    
def get_nans_blocks_length(a):
    """
    Returns 1D length of np.nan s block in sequence depth wise (last axis).
    """
    nan_mask = np.isnan(a)
    start_nans_mask = np.concatenate((np.resize(nan_mask[...,0],a.shape[:-1]+(1,)),
                                 np.logical_and(np.logical_not(nan_mask[...,:-1]), nan_mask[...,1:])
                                 ), axis=a.ndim-1)
    stop_nans_mask = np.concatenate((np.logical_and(nan_mask[...,:-1], np.logical_not(nan_mask[...,1:])),
                                np.resize(nan_mask[...,-1], a.shape[:-1]+(1,))
                                ), axis=a.ndim-1)

    start_idxs = np.where(start_nans_mask)
    stop_idxs = np.where(stop_nans_mask)
    return stop_idxs[-1] - start_idxs[-1] + 1


def get_vals_blocks_length(a):
    """
    Returns 1D length of np.nan s block in sequence depth wise (last axis).
    """
    nan_mask = ~np.isnan(a)
    start_nans_mask = np.concatenate((np.resize(nan_mask[...,0],a.shape[:-1]+(1,)),
                                 np.logical_and(np.logical_not(nan_mask[...,:-1]), nan_mask[...,1:])
                                 ), axis=a.ndim-1)
    stop_nans_mask = np.concatenate((np.logical_and(nan_mask[...,:-1], np.logical_not(nan_mask[...,1:])),
                                np.resize(nan_mask[...,-1], a.shape[:-1]+(1,))
                                ), axis=a.ndim-1)

    start_idxs = np.where(start_nans_mask)
    stop_idxs = np.where(stop_nans_mask)
    return stop_idxs[-1] - start_idxs[-1] + 1


#%% stride length additional features

from poser.utils import butter_highpass_filter, butter_lowpass_filter
from scipy.signal import find_peaks
from poser.structural_features import compute_distances
import scipy as sc

from lempel_ziv_complexity import lempel_ziv_complexity

feats = [('FP right ventral', 'x'), 
            ( 'FP left ventral', 'x'),
            ('BP right ventral', 'x'),
            ( 'BP left ventral', 'x'),            
            ]    # ('FHand lat', 'x'), 

ventral_markers = [ 'TB ventral',
                     'snout ventral',
                     'mouth ventral',
                     'hip mid ventral',
]
    

for p in pc.patients:
    pose_estimation = p.pose_estimation.copy()
    raw_pose_estimation = p.raw_pose_estimation.copy()
    
    data = pose_estimation[feats] 
    raw_data = raw_pose_estimation[feats] 
        
    for f in feats:
        #f = feats[2]
        fs=160       
        
        data = data.interpolate()
  
        x = data.loc[:,(f[0],f[1])]    
        #x = butter_highpass_filter(x, 2, fs)   

        
        peaks,_ = find_peaks(x, distance=20, prominence=10)
        troughs,_ = find_peaks(-x, distance=20, prominence=10)
        
        
        try:
            peaks,troughs = ensure_peak_to_trough(peaks,troughs)
        except:
            continue

        
        #plt.figure()
        #plt.plot(x); plt.scatter(peaks, x[peaks],c='r');  plt.scatter(troughs, x[troughs],c='b')
        #plt.title(p.patient_id + '_' + '_'.join(f) )
        
        n = np.min([len(troughs),len(peaks)])
        
        peaks = peaks[:n]
        troughs = troughs[:n]
        
        wavelength_peak = (peaks[1:] - peaks[:-1])/fs 
        wavelength_trough = (troughs[1:] - troughs[:-1] )/fs 
        
        if troughs[0]>peaks[0]:  
            ramp_time = abs(troughs[:-1] - peaks[1:])/fs 
            decay_time = abs(peaks - troughs)/fs 
        else:
            ramp_time = abs(peaks - troughs)/fs 
            decay_time = abs(troughs[1:] - peaks[:-1])/fs 
            
        n = np.min([len(ramp_time),len(decay_time)])
        assymetry_time = abs(ramp_time[:n]-decay_time[:n])
        
        amplitudes = x[peaks].values - x[troughs].values
        
        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'mean_stride_amplitude'] = np.mean(amplitudes)
        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'median_stride_amplitude'] = np.median(amplitudes)
        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'std_stride_amplitude'] = np.std(amplitudes)

        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'mean_stride_frequency'] = np.mean(1/wavelength_peak)
        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'median_stride_frequency'] = np.median(1/wavelength_peak) 
        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'std_stride_frequency'] = np.std(1/wavelength_peak) 

        # find blocks of 
        
        swing_phase = get_nans_blocks_length(raw_data[f].values)
        stance_phase = get_vals_blocks_length(raw_data[f].values)

        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'median_swing_phase'] = np.median(swing_phase)/fs
        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'max_swing_phase'] = np.max(swing_phase) /fs
        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'std_swing_phase'] = np.std(swing_phase) /fs

        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'median_stance_phase'] = np.median(stance_phase)/fs
        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'max_stance_phase'] = np.max(stance_phase) /fs
        features_df.loc[p.patient_id, f[0] + '_' + f[1] + '_' + 'std_stance_phase'] = np.std(stance_phase) /fs

        
        

import itertools

def compute_distances(data, features):
    """ Compute distances between defined markers """

    if not any(isinstance(el, list) for el in features):
        features = list(itertools.combinations(features, 2))
        features = [list(feat) for feat in features]

    distances_df = pd.DataFrame(index = data.index)

    for f in features:
        distances_df['distance_'+'_'.join(f)] = distance(data, f)
            
    return distances_df 


def distance(data, feature):
    """ computing pairwise distance """
    
    delta_x = data.loc[:, pd.IndexSlice[feature[0], 'x']] - data.loc[:, pd.IndexSlice[feature[1], 'x']]
    dist = np.sqrt(delta_x**2)
    
    return dist


#%% save features


from datetime import datetime

now = datetime.now() # current date and time
             
features_df.to_csv('./outputs/feature_matrix.csv')

#%% example plots


#data.plot()
#plt.title(p.patient_id)
from scipy import signal

for p in pc.patients:

    pose_estimation = p.pose_estimation
    data = pose_estimation[feats]  
    fs = 160
    data = data.interpolate()
    
    data.plot()
    
    plt.figure()
    for f in feats:
        plt.plot(pose_estimation.loc[:,('time','')],data.loc[:,(f[0],f[1])])   
    
    plt.xlabel('Time (s)')
    plt.ylabel('x-marker position')  
    plt.title(p.patient_id)
    plt.savefig('./outputs/figure_plots/example_paws_x_{}.svg'.format(p.patient_id))
    data.to_csv('./outputs/figure_plots/example_paw_x_plot_data_{}.csv'.format(p.patient_id))

    plt.figure()
    freqs = []; amps = []
    for f in feats:
        x = data.loc[:,(f[0],f[1])]    
        x = np.pad(x,(int(fs/2),int(fs/2)),mode='symmetric')   
        
        min_freq = 0
        max_freq = 10
    
        amplitudes = []
        frequencies = []    

        frequency, amplitude = signal.welch(x, fs, nperseg=len(x))
        plt.plot(frequency[frequency<10], np.sqrt(amplitude[frequency<10]))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('sqrt(PSD)')
        
        freqs.append(pd.Series(frequency))
        amps.append(pd.Series(amplitude))
        
    psd_data = pd.concat([pd.Series(frequency),pd.concat(amps,axis=1)],axis=1)
    
    plt.savefig('./outputs/figure_plots/example_paws_psd_{}.svg'.format(p.patient_id))    
    psd_data.to_csv('./outputs/figure_plots/example_psd_plot_data_{}.csv'.format(p.patient_id))
    
    

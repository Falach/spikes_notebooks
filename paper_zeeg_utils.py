import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from depth_utils import get_metrics, calc_features_before_split, calc_features_after_split, channel_feat
import joblib
from mne_features.univariate import get_univariate_funcs, compute_pow_freq_bands
import mne_features
from mne_features.feature_extraction import extract_features
import antropy as ant
import scipy.stats as sp_stats
import time

# General params
sr = 1000
mtl_path = 'C:\\clean_zeeg\\P%s_mtl_clean.fif'
tlv_subjects = ['013', '017', '018', '025', '38', '39', '44', '46', '47', '48', '49', '51', '53', '54', '55', '56', '57']
bonn_subjects = ['707', '708', '709', '710', '711', '712', '713', '714', '715', '723', '724', '728', '731', '733', '734', '735', '737', '744', '746', '752']
milan_subjects = ['801', '802', '804', '805', '807', '809', '810', '812', '813', '814', '815', '816', '817', '818']
all_subjects = tlv_subjects + bonn_subjects + milan_subjects
depth_channels = ['RAH1', 'LAH1', 'RA1', 'LA1', 'LEC1', 'REC1', 'RPHG1', 'LPHG1', 'RMH1', 'LMH1', 'LH1', 'RH1', 'RA3'
                  'RAH2', 'LAH2', 'RA2', 'LA2', 'LEC2', 'REC2', 'RPHG2', 'LPHG2', 'RMH2', 'LMH2', 'LH2', 'RH2']
scalp_channels = ['C3', 'C4', 'PZ', 'EOG1', 'EOG2', 'F4', 'P4', 'F10', 'T10', 'F3', 'P3', 'F9', 'T9', 'CZ', 'P4', 'F8',
                  'T8', 'P8', 'O1', 'O2', 'T5', 'C6', 'P6', 'F7', 'C5', 'P5', 'FZ']

# model = joblib.load(r'C:\repos\depth_ieds\paper\model_V6_90_10.pkl')
model = joblib.load(r'C:\repos\depth_ieds\paper\lgbm_full_f15_s25_b_V5.pkl')
depth_model, feature_names = model['model'], model['features']
window_size = 250  # ms

def extract_epochs_top_features(epochs, subj, sr):
    # Record the start time
    start_time = time.time()

    mobility, complexity = ant.hjorth_params(epochs, axis=1)
    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'kurtosis': sp_stats.kurtosis(epochs, axis=1),
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity,
        'ptp_amp': np.ptp(epochs, axis=1),
        'samp_entropy': np.apply_along_axis(ant.sample_entropy, axis=1, arr=epochs)
    }

    kaiser = mne_features.univariate.compute_teager_kaiser_energy(np.array(epochs))

    # Reshape the list into a 2D array with 12 columns (each row will have 12 values)
    reshaped_list = np.array(kaiser).reshape(-1, 12)

    # Create the DataFrame
    X_new = pd.DataFrame(reshaped_list)
    # rename columns
    X_new.columns = [
        f'teager_kaiser_energy_{i}_mean' if j % 2 == 0 else f'teager_kaiser_energy_{i}_std'
        for i in range(6) for j in range(2)
    ]

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    feat = pd.concat([feat, X_new], axis=1)

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Feature extraction took {:.2f} seconds".format(elapsed_time))
    return feat


def extract_epochs_features_mne(epochs, subj, sr):
    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
    }

    selected_funcs = get_univariate_funcs(sr)
    selected_funcs.pop('spect_edge_freq', None)
    bands_dict = {'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30), 'gamma': (30, 100), 'fast': (100, 300)}
    params = {'pow_freq_bands__freq_bands': bands_dict, 'pow_freq_bands__ratios': 'all', 'pow_freq_bands__psd_method': 'multitaper',
              'energy_freq_bands__freq_bands': bands_dict}
    X_new = extract_features(np.array(epochs)[:, np.newaxis, :], sr, selected_funcs, funcs_params=params, return_as_df=True)
    X_new['abspow'] = compute_pow_freq_bands(sr, np.array(epochs), {'total': (0.1, 500)}, False, psd_method='multitaper')
    # rename columns
    names = []
    for name in X_new.columns:
        if type(name) is tuple:
            if name[1] == 'ch0':
                names.append(name[0])
            else:
                names.append(name[0] + '_' + name[1].replace('ch0_', ''))
        else:
            names.append(name)

    X_new.columns = names

    # add ratios between bands
    X_new['energy_freq_bands_ab'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_beta']
    X_new['energy_freq_bands_ag'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_gamma']
    X_new['energy_freq_bands_as'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_sigma']
    X_new['energy_freq_bands_af'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_at'] = X_new['energy_freq_bands_alpha'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_bt'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_bs'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_sigma']
    X_new['energy_freq_bands_bg'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_gamma']
    X_new['energy_freq_bands_bf'] = X_new['energy_freq_bands_beta'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_st'] = X_new['energy_freq_bands_sigma'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_sg'] = X_new['energy_freq_bands_sigma'] / X_new['energy_freq_bands_gamma']
    X_new['energy_freq_bands_sf'] = X_new['energy_freq_bands_sigma'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_gt'] = X_new['energy_freq_bands_gamma'] / X_new['energy_freq_bands_theta']
    X_new['energy_freq_bands_gf'] = X_new['energy_freq_bands_gamma'] / X_new['energy_freq_bands_fast']
    X_new['energy_freq_bands_ft'] = X_new['energy_freq_bands_fast'] / X_new['energy_freq_bands_theta']

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    feat = pd.concat([feat, X_new], axis=1)

    return feat


def raw_chan_to_feat(raw, chan, subj, top=True):
    epochs = []
    if chan not in raw.ch_names:
        return pd.DataFrame()
    chan_raw = raw.copy().pick([chan]).get_data(reject_by_annotation='NaN').flatten()
    # normalize chan
    chan_norm = (chan_raw - np.nanmean(chan_raw)) / np.nanstd(chan_raw)
    # run on all 250ms epochs (exclude last second)
    for i in range(0, len(chan_norm) - 4 * window_size, window_size):
        if not np.isnan(chan_norm[i: i + window_size]).any():
            epochs.append(chan_norm[i: i + window_size])

    if top:
        curr_feat = extract_epochs_top_features(epochs, subj, raw.info['sfreq'])
    else:
        curr_feat = extract_epochs_features_mne(epochs, subj, raw.info['sfreq'])
    chan_feat = {
        'chan_name': chan,
        'chan_ptp': np.ptp(chan_norm[~np.isnan(chan_norm)]),
        'chan_skew': sp_stats.skew(chan_norm[~np.isnan(chan_norm)]),
        'chan_kurt': sp_stats.kurtosis(chan_norm[~np.isnan(chan_norm)]),
    }

    for feat in chan_feat.keys():
        curr_feat[feat] = chan_feat[feat]

    return curr_feat

def chan_features(subjects, chan):
    chan_feat = {}
    for subj in subjects:
        raw = mne.io.read_raw(mtl_path % subj)
        if chan not in raw.ch_names:
            chan_feat[subj] = {}
        else:
            chan_raw = raw.copy().pick([chan]).get_data(reject_by_annotation='NaN').flatten()
            # normalize chan
            chan_norm = (chan_raw - np.nanmean(chan_raw)) / np.nanstd(chan_raw)
            chan_feat[subj] = {
                'chan_name': chan,
                'chan_ptp': np.ptp(chan_norm[~np.isnan(chan_norm)]),
                'chan_skew': sp_stats.skew(chan_norm[~np.isnan(chan_norm)]),
                'chan_kurt': sp_stats.kurtosis(chan_norm[~np.isnan(chan_norm)]),
            }

    return chan_feat

def map_nan_index(edf):
    raw = mne.io.read_raw(edf)
    raw_data = raw.pick_channels([raw.ch_names[0]])
    if raw_data.info['sfreq'] != sr:
        raw_data.resample(sr)
    raw_data = raw_data.get_data(reject_by_annotation='NaN')[0]
    map = []

    for j, i in enumerate(range(0, len(raw_data), window_size)):
        curr_block = raw_data[i: i + window_size]
        if i + window_size < len(raw_data):
            if not np.isnan(curr_block).any():
                map.append(j)
            else:
                print('nan')
    return map
# index_map = map_nan_index('D:\\TLV\\%s_clean_mtl_annot.fif' % '025')

def get_depth_pred_unbalanced(subjects, threshold=0.8):
    y_all = {}
    for subj in subjects:
        raw = mne.io.read_raw(mtl_path % subj)
        curr_chans = [chan for chan in raw.ch_names if chan in depth_channels]
        y_curr = None
        for chan in curr_chans:
            curr_feat = raw_chan_to_feat(raw, chan, subj)
            predictions = depth_model.predict_proba(curr_feat[feature_names])
            print(sum((predictions[:, 1] >= threshold).astype(int)), chan)
            if y_curr is None:
                y_curr = (predictions[:, 1] >= threshold).astype(int)
            else:
                y_curr += (predictions[:, 1] >= threshold).astype(int)

        # at least 2 channels should be above threshold
        y_curr[y_curr == 1] = 0
        y_curr[y_curr > 1] = 1
        y_all[subj] = y_curr
        joblib.dump(y_all, 'y_balanced_51_atleast2.pkl')
        print("finish " + subj)

    return y_all


def get_depth_pred_laterlity(subjects):
    y_all = {}
    for subj in subjects:
        raw = mne.io.read_raw(mtl_path % subj)
        for side in ['L', 'R']:
            curr_chans = [chan for chan in raw.ch_names if chan in depth_channels and chan[0] == side]
            if len(curr_chans) == 0:
                continue
            # get only one channel from each location
            min_indexes = {}
            for item in curr_chans:
                prefix = item[:-1]
                index = int(item[-1])
                if prefix not in min_indexes or index < int(min_indexes[prefix][-1][-1]):
                    min_indexes[prefix] = item
            y_curr = None
            for chan in min_indexes.values():
                curr_feat = raw_chan_to_feat(raw, chan, subj)
                predictions = depth_model.predict_proba(curr_feat[feature_names])
                print(sum((predictions[:, 1] >= 0.8).astype(int)), chan)
                if y_curr is None:
                    y_curr = (predictions[:, 1] >= 0.8).astype(int)
                else:
                    y_curr += (predictions[:, 1] >= 0.8).astype(int)

            y_curr[y_curr > 1] = 1
            if subj not in y_all:
                y_all[subj] = {side: y_curr}
            else:
                y_all[subj][side] = y_curr
        joblib.dump(y_all, 'lateral_y.pkl')

    return y_all


def get_all_features_per_chan(chan, subjects):
    all_features = {}
    for subj in subjects:
        raw = mne.io.read_raw(mtl_path % subj)
        curr_feat = raw_chan_to_feat(raw, chan, subj, top=False)
        all_features[subj] = curr_feat
        joblib.dump(all_features, f'{chan}_mne_51.pkl')

    return all_features

def save_dicts():
    subj_data = {}
    y_all = joblib.load(r'y_balanced_51_atleast2.pkl')
    eog1 = joblib.load(r'EOG1_mne_51.pkl')
    eog2 = joblib.load(r'EOG2_mne_51.pkl')
    eog0 = joblib.load(r'EOG_mne_51.pkl')
    fix_1 = joblib.load(r'eog1_chan_fix.pkl')
    fix_2 = joblib.load(r'eog2_chan_fix.pkl')
    fix_0 = joblib.load(r'eog_chan_fix.pkl')

    for subj in [x for x in tlv_subjects+bonn_subjects if x not in ['025', '707']]:
        print(subj)
        eog1_subj = eog1[subj]
        eog1_subj['chan_ptp'] = fix_1[subj]['chan_ptp']
        eog1_subj['chan_skew'] = fix_1[subj]['chan_skew']
        eog1_subj['chan_kurt'] = fix_1[subj]['chan_kurt']
        eog2_subj = eog2[subj]
        eog2_subj['chan_ptp'] = fix_2[subj]['chan_ptp']
        eog2_subj['chan_skew'] = fix_2[subj]['chan_skew']
        eog2_subj['chan_kurt'] = fix_2[subj]['chan_kurt']
        subj_data[subj] = {'eog1': eog1_subj, 'eog2': eog2_subj, 'y': y_all[subj]}

    for subj in milan_subjects:
        print(subj)
        eog0_subj = eog0[subj]
        eog0_subj['chan_ptp'] = fix_0[subj]['chan_ptp']
        eog0_subj['chan_skew'] = fix_0[subj]['chan_skew']
        eog0_subj['chan_kurt'] = fix_0[subj]['chan_kurt']
        subj_data[subj] = {'eog1': eog0_subj, 'eog2': eog0_subj, 'y': y_all[subj]}

    #025
    eog1_subj = eog1['025']
    eog1_subj['chan_ptp'] = fix_1['025']['chan_ptp']
    eog1_subj['chan_skew'] = fix_1['025']['chan_skew']
    eog1_subj['chan_kurt'] = fix_1['025']['chan_kurt']
    subj_data['025'] = {'eog1': eog1_subj, 'eog2': eog1_subj, 'y': y_all['025']}

    #707
    eog2_subj = eog2['707']
    eog2_subj['chan_ptp'] = fix_2['707']['chan_ptp']
    eog2_subj['chan_skew'] = fix_2['707']['chan_skew']
    eog2_subj['chan_kurt'] = fix_2['707']['chan_kurt']
    subj_data['707'] = {'eog1': eog2_subj, 'eog2': eog2_subj, 'y': y_all['707']}

    joblib.dump(subj_data, 'subj_data_final_yb.pkl')
    return subj_data


# y = get_depth_pred_unbalanced(['39'])
# joblib.dump(y, 'y_18.pkl')
# eog1 = raw_chan_to_feat(mne.io.read_raw(mtl_path % '018'), 'EOG1', '018')
# eog2 = raw_chan_to_feat(mne.io.read_raw(mtl_path % '018'), 'EOG2', '018')
# joblib.dump({1: eog1, 2: eog2}, 'eog_18.pkl')

# y = get_depth_pred_unbalanced(all_subjects, threshold=0.8)
# eog1_all = get_all_features_per_chan('EOG1', all_subjects)
# joblib.dump(eog1_all, 'eog1_mne_51.pkl')
# eog2_all = get_all_features_per_chan('EOG2', ['46', '47', '48', '49', '51', '53', '54', '55', '56', '57'] + bonn_subjects + milan_subjects)
# joblib.dump(eog2_all, 'eog2_mne_51.pkl')
# eog_all = get_all_features_per_chan('EOG', all_subjects)
# joblib.dump(eog_all, 'eog_mne_51.pkl')
# lateral = get_depth_pred_laterlity(all_subjects)
# joblib.dump(lateral, 'lateral_y.pkl')

# fix the nan values in the chan features
# chan_feat1 = chan_features(all_subjects, 'EOG1')
# joblib.dump(chan_feat1, f'eog1_chan_fix.pkl')
# chan_feat2 = chan_features(all_subjects, 'EOG2')
# joblib.dump(chan_feat2, f'eog2_chan_fix.pkl')
# chan_feat = chan_features(all_subjects, 'EOG')
# joblib.dump(chan_feat, f'eog_chan_fix.pkl')

# create nice dicts with everything




final_dict = save_dicts()

# laterality = joblib.load('laterality_results.pkl')
# laterality_avg = {}
# for key, value in laterality.items():
#     laterality_avg[key] = value.iloc[5,0]
# print('done')

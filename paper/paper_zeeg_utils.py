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
depth_channels = ['RAH1', 'LAH1', 'RA1', 'LA1', 'LEC1', 'REC1', 'RPHG1', 'LPHG1', 'RMH1', 'LMH1', 'LH1', 'RH1', 'RA3',
                  'RAH2', 'LAH2', 'RA2', 'LA2', 'LEC2', 'REC2', 'RPHG2', 'LPHG2', 'RMH2', 'LMH2', 'LH2', 'RH2']
scalp_channels = ['C3', 'C4', 'PZ', 'EOG1', 'EOG2', 'F4', 'P4', 'F10', 'T10', 'F3', 'P3', 'F9', 'T9', 'CZ', 'P4', 'F8',
                  'T8', 'P8', 'O1', 'O2', 'T5', 'C6', 'P6', 'F7', 'C5', 'P5', 'FZ']
id_name_map = {'44': 'AO0', '46': 'MM1', '47': 'SS1', '48': 'TA1', '49': 'MF7', '51': 'RS4', '53': 'IA2', '54': 'MA5',
             '55': 'NG6', '56': 'DR1', '57': 'MR9'}

model = joblib.load(r'C:\repos\depth_ieds\paper\model_V6_90_10.pkl')
# model = joblib.load(r'C:\repos\depth_ieds\paper\lgbm_full_f15_s25_b_V5.pkl')
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


def get_subj_features_zeeg_fast(epochs, subj, sr):
    data = np.array(epochs)

    ptp_amp = mne_features.univariate.compute_ptp_amp(data)
    hjorth_mobility = mne_features.univariate.compute_hjorth_mobility(data)
    hjorth_complexity = mne_features.univariate.compute_hjorth_complexity(data)
    samp_entropy = mne_features.univariate.compute_samp_entropy(data, emb=2, metric='chebyshev')
    compute_spect_slope = mne_features.univariate.compute_spect_slope(
        sr, data, fmin=0.1, fmax=50, with_intercept=True, psd_method='welch', psd_params=None
    )

    # Compute power in specified frequency bands with normalization enabled and no ratios
    bands_dict = {'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30), 'gamma': (30, 100), 'fast': (100, 300)}

    energy_freq_bands = mne_features.univariate.compute_energy_freq_bands(sr, data, freq_bands=bands_dict)
    app_entropy = mne_features.univariate.compute_app_entropy(data, emb=2, metric='chebyshev')
    decorr_time = mne_features.univariate.compute_decorr_time(sr, data)
    higuchi_fd = mne_features.univariate.compute_higuchi_fd(data, kmax=10)
    hjorth_complexity_spect = mne_features.univariate.compute_hjorth_complexity_spect(sr, data, normalize=False, psd_method='welch', psd_params=None)
    hjorth_mobility_spect = mne_features.univariate.compute_hjorth_mobility_spect(sr, data, normalize=False, psd_method='welch', psd_params=None)
    hurst_exp = mne_features.univariate.compute_hurst_exp(data)
    katz_fd = mne_features.univariate.compute_katz_fd(data)
    kurtosis = mne_features.univariate.compute_kurtosis(data)
    line_length = mne_features.univariate.compute_line_length(data)
    mean = mne_features.univariate.compute_mean(data)
    quantile = mne_features.univariate.compute_quantile(data, q=0.75)
    rms = mne_features.univariate.compute_rms(data)
    skewness = mne_features.univariate.compute_skewness(data)
    spect_entropy = mne_features.univariate.compute_spect_entropy(sr, data, psd_method='welch', psd_params=None)
    std = mne_features.univariate.compute_std(data)
    svd_entropy = mne_features.univariate.compute_svd_entropy(data, tau=2, emb=10)
    svd_fisher_info = mne_features.univariate.compute_svd_fisher_info(data, tau=2, emb=10)
    variance = mne_features.univariate.compute_variance(data)
    zero_crossings = mne_features.univariate.compute_zero_crossings(data, threshold=2.220446049250313e-16)
    abspow = mne_features.univariate.compute_pow_freq_bands(sr, data, {'total': (0.1, 500)}, False,
                                                            psd_method='multitaper')
    # Stack the computed feature arrays into columns for easier DataFrame creation
    stacked_arrays = np.column_stack(
        (hjorth_mobility, hjorth_complexity, samp_entropy, ptp_amp, app_entropy, decorr_time, higuchi_fd,
         hjorth_complexity_spect, hjorth_mobility_spect, hurst_exp, katz_fd, kurtosis, line_length, mean,
         quantile, rms, skewness, spect_entropy, std, svd_entropy, svd_fisher_info,
         variance, zero_crossings, abspow))

    df = pd.DataFrame(stacked_arrays)

    df.columns = ['hjorth_mobility', 'hjorth_complexity', 'samp_entropy', 'ptp_amp', 'app_entropy', 'decorr_time',
                  'higuchi_fd', 'hjorth_complexity_spect', 'hjorth_mobility_spect', 'hurst_exp', 'katz_fd', 'kurtosis',
                  'line_length', 'mean', 'quantile', 'rms', 'skewness', 'spect_entropy',
                  'std', 'svd_entropy', 'svd_fisher_info', 'variance', 'zero_crossings', 'abspow_']

    # Add metadata columns for subject, channel name, epoch label, and epoch ID
    df['subj'] = np.full(df.shape[0], subj)  # `subj` should be defined or passed into the function
    df['epoch_id'] = list(range(0, len(epochs)))  # Assigns a unique ID for each epoch

    kaiser = mne_features.univariate.compute_teager_kaiser_energy(data)
    kaiser_df = pd.DataFrame(
        np.array(kaiser).reshape(-1, 12),
        columns=['teager_kaiser_energy_0_mean', 'teager_kaiser_energy_0_std',
                 'teager_kaiser_energy_1_mean', 'teager_kaiser_energy_1_std',
                 'teager_kaiser_energy_2_mean', 'teager_kaiser_energy_2_std',
                 'teager_kaiser_energy_3_mean', 'teager_kaiser_energy_3_std',
                 'teager_kaiser_energy_4_mean', 'teager_kaiser_energy_4_std',
                 'teager_kaiser_energy_5_mean', 'teager_kaiser_energy_5_std']
    )

    pow_freq_bands = mne_features.univariate.compute_pow_freq_bands(data=data, sfreq=sr, freq_bands=bands_dict,
                                                                    normalize=True, ratios=None, ratios_triu=False,
                                                                    psd_method='multitaper', log=False, psd_params=None)
    pow_freq_bands_df = pd.DataFrame(np.array(pow_freq_bands).reshape(-1, 6),
                                     columns=['pow_freq_bands_theta', 'pow_freq_bands_alpha', 'pow_freq_bands_sigma',
                                              'pow_freq_bands_beta', 'pow_freq_bands_gamma', 'pow_freq_bands_fast'])

    for band1 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
        for band2 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
            if band1 == band2:
                continue
            else:
                pow_freq_bands_df[f'pow_freq_bands_{band1}/{band2}'] = pow_freq_bands_df[f'pow_freq_bands_{band1}'] / \
                                                                       pow_freq_bands_df[f'pow_freq_bands_{band2}']

    # Create DataFrame for spectral slope components (intercept, slope, MSE, and R2)
    spect_slope = pd.DataFrame(
        np.array(compute_spect_slope).reshape(-1, 4),
        columns=['spect_slope_intercept', 'spect_slope_slope', 'spect_slope_MSE', 'spect_slope_R2']
    )

    # Create DataFrame for energy frequency bands with each band as a separate column
    energy_freq_bands = pd.DataFrame(
        np.array(energy_freq_bands).reshape(-1, 6),
        columns=['energy_freq_bands_theta', 'energy_freq_bands_alpha', 'energy_freq_bands_sigma',
                 'energy_freq_bands_beta', 'energy_freq_bands_gamma', 'energy_freq_bands_fast']
    )
    # add ratios between bands
    energy_freq_bands['energy_freq_bands_gf'] = energy_freq_bands['energy_freq_bands_gamma'] / energy_freq_bands[
        'energy_freq_bands_fast']
    energy_freq_bands['energy_freq_bands_bg'] = energy_freq_bands['energy_freq_bands_beta'] / energy_freq_bands[
        'energy_freq_bands_gamma']

    wavelet_coef_energy = mne_features.univariate.compute_wavelet_coef_energy(data, wavelet_name='db4')
    wavelet_coef_energy_df = pd.DataFrame(
        np.array(wavelet_coef_energy).reshape(-1, 5),
        columns=['wavelet_coef_energy_0', 'wavelet_coef_energy_1', 'wavelet_coef_energy_2', 'wavelet_coef_energy_3',
                 'wavelet_coef_energy_4'])

    # Concatenate all feature DataFrames (df, spect_slope, pow_freq_bands, energy_freq_bands, kaiser_df) side by side
    subj_features = pd.concat(
        [df, spect_slope, energy_freq_bands, kaiser_df, pow_freq_bands_df, wavelet_coef_energy_df], axis=1)

    # add ratios between bands
    for band1 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
        for band2 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
            if band1 == band2:
                continue
            else:
                subj_features[f'energy_freq_bands_{band1[0]}{band2[0]}'] = subj_features[f'energy_freq_bands_{band1}'] / \
                                                                           subj_features[f'energy_freq_bands_{band2}']
    subj_features = subj_features[
        ['subj', 'epoch_id', 'app_entropy', 'decorr_time', 'energy_freq_bands_theta', 'energy_freq_bands_alpha',
         'energy_freq_bands_sigma', 'energy_freq_bands_beta', 'energy_freq_bands_gamma', 'energy_freq_bands_fast',
         'higuchi_fd', 'hjorth_complexity', 'hjorth_complexity_spect', 'hjorth_mobility', 'hjorth_mobility_spect',
         'hurst_exp', 'katz_fd', 'kurtosis', 'line_length', 'mean', 'pow_freq_bands_theta', 'pow_freq_bands_alpha',
         'pow_freq_bands_sigma', 'pow_freq_bands_beta', 'pow_freq_bands_gamma', 'pow_freq_bands_fast',
         'pow_freq_bands_theta/alpha', 'pow_freq_bands_theta/sigma', 'pow_freq_bands_theta/beta',
         'pow_freq_bands_theta/gamma', 'pow_freq_bands_theta/fast', 'pow_freq_bands_alpha/theta',
         'pow_freq_bands_alpha/sigma', 'pow_freq_bands_alpha/beta', 'pow_freq_bands_alpha/gamma',
         'pow_freq_bands_alpha/fast', 'pow_freq_bands_sigma/theta', 'pow_freq_bands_sigma/alpha',
         'pow_freq_bands_sigma/beta', 'pow_freq_bands_sigma/gamma', 'pow_freq_bands_sigma/fast',
         'pow_freq_bands_beta/theta', 'pow_freq_bands_beta/alpha', 'pow_freq_bands_beta/sigma',
         'pow_freq_bands_beta/gamma', 'pow_freq_bands_beta/fast', 'pow_freq_bands_gamma/theta',
         'pow_freq_bands_gamma/alpha', 'pow_freq_bands_gamma/sigma', 'pow_freq_bands_gamma/beta',
         'pow_freq_bands_gamma/fast', 'pow_freq_bands_fast/theta', 'pow_freq_bands_fast/alpha',
         'pow_freq_bands_fast/sigma', 'pow_freq_bands_fast/beta', 'pow_freq_bands_fast/gamma', 'ptp_amp', 'quantile',
         'rms', 'samp_entropy', 'skewness', 'spect_entropy', 'spect_slope_intercept', 'spect_slope_slope',
         'spect_slope_MSE', 'spect_slope_R2', 'std', 'svd_entropy', 'svd_fisher_info', 'teager_kaiser_energy_0_mean',
         'teager_kaiser_energy_0_std', 'teager_kaiser_energy_1_mean', 'teager_kaiser_energy_1_std',
         'teager_kaiser_energy_2_mean', 'teager_kaiser_energy_2_std', 'teager_kaiser_energy_3_mean',
         'teager_kaiser_energy_3_std', 'teager_kaiser_energy_4_mean', 'teager_kaiser_energy_4_std',
         'teager_kaiser_energy_5_mean', 'teager_kaiser_energy_5_std', 'variance', 'wavelet_coef_energy_0',
         'wavelet_coef_energy_1', 'wavelet_coef_energy_2', 'wavelet_coef_energy_3', 'wavelet_coef_energy_4',
         'zero_crossings', 'abspow_', 'energy_freq_bands_ab', 'energy_freq_bands_ag', 'energy_freq_bands_as',
         'energy_freq_bands_af', 'energy_freq_bands_at', 'energy_freq_bands_bt', 'energy_freq_bands_bs',
         'energy_freq_bands_bg', 'energy_freq_bands_bf', 'energy_freq_bands_st', 'energy_freq_bands_sg',
         'energy_freq_bands_sf', 'energy_freq_bands_gt', 'energy_freq_bands_gf', 'energy_freq_bands_ft']]
    return subj_features


def moving_average_nan(channel, window_size):
    """Applies a moving average while ignoring NaNs and keeping them in place."""
    nan_mask = np.isnan(channel)

    # Replace NaNs with zero (or any placeholder) for convolution
    valid_mask = ~nan_mask
    smoothed = np.convolve(np.where(valid_mask, channel, 0), np.ones(window_size) / window_size, mode='same')
    normalizing = np.convolve(valid_mask.astype(float), np.ones(window_size), mode='same')

    # Normalize and restore NaNs
    smoothed = smoothed / normalizing
    smoothed[nan_mask] = np.nan  # Restore NaNs

    return smoothed

def raw_chan_to_feat(raw, chan, subj, top=False, sma=False):
    epochs = []
    if chan not in raw.ch_names:
        return pd.DataFrame()

    chan_raw = raw.copy().pick([chan]).get_data(reject_by_annotation='NaN').flatten()
    if sma:
        # Apply smoothing while keeping NaNs
        chan_raw = chan_raw - moving_average_nan(chan_raw, int(raw.info['sfreq'] * 2))
    # normalize chan
    chan_norm = (chan_raw - np.nanmean(chan_raw)) / np.nanstd(chan_raw)
    # run on all 250ms epochs (exclude last second)
    for i in range(0, len(chan_norm) - 4 * window_size, window_size):
        if not np.isnan(chan_norm[i: i + window_size]).any():
            epochs.append(chan_norm[i: i + window_size])

    if top:
        curr_feat = extract_epochs_top_features(epochs, subj, raw.info['sfreq'])
    else:
        curr_feat = get_subj_features_zeeg_fast(epochs, subj, raw.info['sfreq'])
    chan_feat = {
        'chan_name': chan,
        'chan_ptp': np.ptp(chan_norm[~np.isnan(chan_norm)]),
        'chan_skew': sp_stats.skew(chan_norm[~np.isnan(chan_norm)]),
        'chan_kurt': sp_stats.kurtosis(chan_norm[~np.isnan(chan_norm)]),
    }

    for feat in chan_feat.keys():
        curr_feat[feat] = chan_feat[feat]

    return curr_feat


def raw_chan_to_feat_test(raw, chan, subj):
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


    curr_feat = get_subj_features_zeeg_fast(epochs, raw.info['sfreq'], chan, subj)
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

#TODO: run this function properly
def get_depth_pred_unbalanced(subjects, out_filename, threshold=0.8, min_channels=2, raw_file=None):
    y_all = {}
    for subj in subjects:
        # TODO: only for 54!!!
        if subj == '54':
            print("read NEW 54!!!")
            raw = mne.io.read_raw(r"C:\clean_zeeg\P54_mtl_clean_NEW.fif")
        else:
            if raw_file is None:
                raw = mne.io.read_raw(mtl_path % subj)
            else:
                raw = mne.io.read_raw(raw_file % subj)
        curr_chans = [chan for chan in raw.ch_names if chan in depth_channels]
        y_curr = None
        for chan in curr_chans:
            curr_feat = raw_chan_to_feat(raw, chan, subj, top=True)
            predictions = depth_model.predict_proba(curr_feat[feature_names])
            print(sum((predictions[:, 1] >= threshold).astype(int)), chan)
            if y_curr is None:
                y_curr = (predictions[:, 1] >= threshold).astype(int)
            else:
                y_curr += (predictions[:, 1] >= threshold).astype(int)

        # at least X channels should be above threshold
        y_curr[y_curr <= min_channels - 1] = 0
        y_curr[y_curr > min_channels - 1] = 1
        y_all[subj] = y_curr
        joblib.dump(y_all, out_filename)
        print("finish " + subj)

    return y_all

# TODO: run this function properly
def get_depth_pred_unbalanced_deepest(subjects, filename, threshold=0.8, min_channels=2):
    y_all = {}
    for subj in subjects:
        raw = mne.io.read_raw(mtl_path % subj)
        curr_chans = [chan for chan in raw.ch_names if chan in depth_channels]
        # get only one deepest channel from each location
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
            print(sum((predictions[:, 1] >= threshold).astype(int)), chan)
            if y_curr is None:
                y_curr = (predictions[:, 1] >= threshold).astype(int)
            else:
                y_curr += (predictions[:, 1] >= threshold).astype(int)

        # at least X channels should be above threshold
        y_curr[y_curr <= min_channels - 1] = 0
        y_curr[y_curr > min_channels - 1] = 1
        y_all[subj] = y_curr
        joblib.dump(y_all, filename)
        print("finish " + subj)


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
        curr_feat = raw_chan_to_feat(raw, chan, subj, top=False, sma=True)
        all_features[subj] = curr_feat
        joblib.dump(all_features, f'{chan}_fast_51_sma.pkl')

    return all_features

def save_dicts_fix(y_file=None):
    subj_data = {}
    if y_file is None:
        y_file = 'y_51_atleast2_fix.pkl'
    y_all = joblib.load(y_file)
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

    joblib.dump(subj_data, f'subj_data_{y_file}.pkl')
    return subj_data

def save_dicts():
    subj_data = {}
    y_all = joblib.load('y_balanced_51_atleast2.pkl')
    eog1 = joblib.load(r'EOG1_fast_51_sma.pkl')
    eog2 = joblib.load(r'EOG2_fast_51_sma.pkl')
    eog0 = joblib.load(r'EOG_fast_51_sma.pkl')

    for subj in [x for x in tlv_subjects+bonn_subjects if x not in ['025', '707']]:
        print(subj)
        eog1_subj = eog1[subj]
        eog2_subj = eog2[subj]
        subj_data[subj] = {'eog1': eog1_subj, 'eog2': eog2_subj, 'y': y_all[subj]}

    for subj in milan_subjects:
        print(subj)
        eog0_subj = eog0[subj]
        subj_data[subj] = {'eog1': eog0_subj, 'eog2': eog0_subj, 'y': y_all[subj]}

    #025
    eog1_subj = eog1['025']
    subj_data['025'] = {'eog1': eog1_subj, 'eog2': eog1_subj, 'y': y_all['025']}

    #707
    eog2_subj = eog2['707']
    subj_data['707'] = {'eog1': eog2_subj, 'eog2': eog2_subj, 'y': y_all['707']}

    joblib.dump(subj_data, f'subj_data_sma_depthb.pkl')
    return subj_data

def nrem_model():
    import mne
    import numpy as np
    subj_data = joblib.load('subj_data_final.pkl')

    # Define subjects
    tlv_subjects = ['013', '017', '018', '025', '38', '39', '44', '46', '47', '48', '51', '53', '54', '56',
                    '57']  # without 49- no scoring, 55- scoring doesnt fit
    mismatch_subjects = {}

    # Process each subject
    for subj in ['44']:
        fif_file = rf"C:\clean_zeeg\P{subj}_mtl_clean.fif"
        hypnogram_file = rf"D:\Ichilov_scoring\P{subj}.txt"

        # Load FIF file
        raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
        # raw = mne.io.read_raw_fif(r"Z:\25. Interictal activities - Rotem\TLV\TLV\54\P54_mtl_filtered.fif", preload=True, verbose=False)
        raw.pick(picks=0)  # Pick only the first channel
        if subj == '55':
            raw.crop(tmax=28355)
        sfreq = raw.info['sfreq']  # Get sampling frequency

        # Load hypnogram
        hypnogram = np.loadtxt(hypnogram_file)

        # Expand hypnogram to 250ms resolution (4 values per second)
        hypnogram_250ms = np.repeat(hypnogram, 4)

        # Compute correct epoch size in samples
        epoch_samples = int(sfreq * 0.25)  # 250ms in samples

        # Get raw data and flatten it (assuming single-channel analysis)
        raw_data = raw.get_data(reject_by_annotation='NaN').flatten()

        # Lists to store valid epochs and corresponding hypnogram indices
        valid_hypnogram_indices = []

        # Run through 250ms epochs
        for i in range(0, len(raw_data) - 4 * epoch_samples, epoch_samples):
            epoch = raw_data[i: i + epoch_samples]

            # Check if the epoch contains NaN
            if not np.isnan(epoch).any():
                valid_hypnogram_indices.append(i // epoch_samples)  # Save the valid hypnogram index 4 times


        # Extract the valid hypnogram values
        valid_hypnogram = hypnogram_250ms[valid_hypnogram_indices]

        # Check for mismatch in number of rows
        expected_epochs = len(subj_data[subj]['y'])  # Expected from subject's dataframe
        actual_epochs = len(valid_hypnogram)  # Hypnogram after NaN filtering

        if expected_epochs != actual_epochs:
            mismatch_subjects[subj] = expected_epochs - actual_epochs  # Store difference

    # Print subjects with mismatches
    if mismatch_subjects:
        print("\nSubjects with dataframe-hypnogram mismatches:")
        for subj, diff in mismatch_subjects.items():
            print(f"P{subj}: Mismatch of {diff} epochs")
    else:
        print("\nAll subjects matched correctly!")


# y = get_depth_pred_unbalanced(['39'])
# joblib.dump(y, 'y_18.pkl')
# eog1 = raw_chan_to_feat(mne.io.read_raw(mtl_path % '018'), 'EOG1', '018')
# eog2 = raw_chan_to_feat(mne.io.read_raw(mtl_path % '018'), 'EOG2', '018')
# joblib.dump({1: eog1, 2: eog2}, 'eog_18.pkl')

# y = get_depth_pred_unbalanced(all_subjects, 'y_balanced_51_atleast1.pkl', threshold=0.8, min_channels=1)
# y = get_depth_pred_unbalanced_deepest(all_subjects, 'y_balanced_51_deepest_atleast1.pkl', threshold=0.8, min_channels=1)
# save_dicts('y_V6_51_atleast1.pkl')
# save_dicts('y_balanced_51_deepest_atleast1.pkl')
# eog1_all = get_all_features_per_chan('EOG1', all_subjects)
# eog2_all = get_all_features_per_chan('EOG2', all_subjects)
# joblib.dump(eog1_all, 'eog1_mne_51.pkl')
# eog2_all = get_all_features_per_chan('EOG2', ['46', '47', '48', '49', '51', '53', '54', '55', '56', '57'] + bonn_subjects + milan_subjects)
# joblib.dump(eog2_all, 'eog2_mne_51.pkl')
# eog_all = get_all_features_per_chan('EOG', all_subjects)
# joblib.dump(eog_all, 'eog_mne_51.pkl')
# lateral = get_depth_pred_laterlity(all_subjects)
# joblib.dump(lateral, 'lateral_y.pkl')


# create nice dicts with everything
# final_dict = save_dicts_fix()
# final_dict = save_dicts()


# laterality = joblib.load('laterality_results.pkl')
# laterality_avg = {}
# for key, value in laterality.items():
#     laterality_avg[key] = value.iloc[5,0]
# print('done')
# nrem_model()

# sma = joblib.load('subj_data_sma.pkl')
# original = joblib.load('subj_data_final.pkl')
# # check if eog 1 is the same length
# print(sma['54']['eog1'].shape)
# print(original['54']['eog1'].shape)

def calc_real_spm():
    subj_data = joblib.load('subj_data_final.pkl')
    ratios = {}
    # calc ratio per subject
    for subj in id_name_map:
        y_subj = subj_data[subj]['y']
        ratios[id_name_map[subj]] = sum(y_subj) / len(y_subj)

    return ratios


def match_y_with_hypno(fif_path, hypno_path, pkl_path, subj_id):
    epoch_samples = 250
    raw = mne.io.read_raw_fif(fif_path, preload=True).pick(picks=0)
    sfreq = raw.info["sfreq"]  # Sampling frequency (e.g., 256 Hz, 512 Hz)

    # Load hypnogram (1 Hz sampling rate)
    hypnogram = np.loadtxt(hypno_path)  # Assuming 1 value per second
    y_data = joblib.load(pkl_path)
    if subj_id not in y_data:
        # get id from name
        x = [key for key, value in id_name_map.items() if value == subj_id][0]
        y_vector = y_data[x]
    else:
        y_vector = y_data[subj_id]  # Extract vector for the subject
    num_seconds = int(raw.n_times / sfreq)  # Total recording length in seconds

    if len(hypnogram) != num_seconds and len(hypnogram) != num_seconds - 1 and len(hypnogram) != num_seconds + 1:
        raise ValueError(f"Mismatch: Hypnogram has {len(hypnogram)} values, expected {num_seconds}.")

    hypnogram_250ms = np.repeat(hypnogram, 4)
    raw_data = raw.get_data(reject_by_annotation='NaN').flatten()
    valid_hypnogram_indices = []

    # Run through 250ms epochs
    for i in range(0, len(raw_data) - 4 * epoch_samples, epoch_samples):
        epoch = raw_data[i: i + epoch_samples]
        # Check if the epoch contains NaN
        if not np.isnan(epoch).any():
            valid_hypnogram_indices.append(i // epoch_samples)  # Save the valid hypnogram index 4 times

    valid_hypnogram = hypnogram_250ms[valid_hypnogram_indices]
    if len(y_vector) != len(valid_hypnogram):
        raise ValueError(f"After removing bad epochs, y_vector and hypnogram length mismatch: "
                         f"{len(y_vector)} vs {len(valid_hypnogram)}")

    return y_vector, valid_hypnogram


def calculate_rates(y, hypno, sfreq=4):
    samples_per_minute = int(60 * sfreq)  # 240 samples per minute

    # --- 1. Find sleep onset to wake period ---
    wake_idx = np.where(hypno == 0)[0]  # Wake indices
    first_sleep = np.min(np.setdiff1d(np.arange(len(hypno)), wake_idx))  # First non-wake
    last_sleep = np.max(np.setdiff1d(np.arange(len(hypno)), wake_idx))   # Last non-wake

    y_sleep = y[first_sleep:last_sleep + 1]  # y values in sleep period
    sleep_minutes = len(y_sleep) / samples_per_minute
    rate_sleep = np.sum(y_sleep == 1) / sleep_minutes  # Events per minute

    # --- 2. NREM Rate ---
    nrem_idx = np.where((hypno == 2) | (hypno == 3))  # NREM includes Stage 2 & 3
    y_nrem = y[nrem_idx]
    nrem_minutes = len(y_nrem) / samples_per_minute
    rate_nrem = np.sum(y_nrem == 1) / nrem_minutes

    # --- 3. First Hour of NREM ---
    first_hour_samples = 60 * samples_per_minute  # Samples in first hour
    first_hour_nrem_idx = nrem_idx[0][:first_hour_samples]  # Get first hour of NREM
    y_nrem_1h = y[first_hour_nrem_idx]
    rate_nrem_1h = np.sum(y_nrem_1h == 1) / 60  # Normalize to per-minute rate

    return rate_sleep, rate_nrem, rate_nrem_1h


def calculate_rates_restricted(y, hypno, sfreq=4, merge_window_sec=1, minutes=60):
    epoch_duration = 1 / sfreq  # Each sample is 250ms, so 4 samples per second
    samples_per_minute = int(60 * sfreq)  # 240 samples per minute
    merge_samples = int(merge_window_sec * sfreq)  # How many samples to check

    def count_detections(y_vector, merge_samples):
        """Counts unique detections based on the merge restriction."""
        if merge_samples == 0:
            return np.sum(y_vector)
        detections = np.where(y_vector == 1)[0]  # Indices of detected events
        if len(detections) == 0:
            return 0
        unique_detections = [detections[0]]  # Always count the first detection

        # Iterate over detections and ensure a merge window
        for idx in detections[1:]:
            if idx - unique_detections[-1] >= merge_samples:
                unique_detections.append(idx)

        return len(unique_detections)

    def count_density(y_vector, merge_samples):
        """Calculates the average distance between unique detections in y_vector.
        If merge_samples == 0, considers every '1' as a detection."""
        # Find detection indices
        detections = np.where(y_vector == 1)[0]
        if len(detections) <= 1:
            return 0  # No average distance if 0 or 1 detection

        # Merge nearby detections if needed
        if merge_samples > 0:
            unique_detections = [detections[0]]
            for idx in detections[1:]:
                if idx - unique_detections[-1] >= merge_samples:
                    unique_detections.append(idx)
        else:
            unique_detections = detections

        # Calculate distances between consecutive detections
        distances = np.diff(unique_detections)
        avg_distance = np.mean(distances) if len(distances) > 0 else 0
        # make it as seconds
        return avg_distance / 4

    # --- 1. Find sleep onset to wake period ---
    wake_idx = np.where(hypno == 0)[0]  # Wake indices
    if np.setdiff1d(np.arange(len(hypno)), wake_idx).size == 0:
        return 0, 0, 0, 60, 0  # No sleep detected
    first_sleep = np.min(np.setdiff1d(np.arange(len(hypno)), wake_idx))  # First non-wake
    last_sleep = np.max(np.setdiff1d(np.arange(len(hypno)), wake_idx))   # Last non-wake

    y_sleep = y[first_sleep:last_sleep + 1]  # y values in sleep period
    sleep_minutes = len(y_sleep) / samples_per_minute
    rate_sleep = count_detections(y_sleep, merge_samples) / sleep_minutes  # Events per minute

    # --- 2. NREM Rate ---
    nrem_idx = np.where((hypno == 2) | (hypno == 3))  # NREM includes Stage 2 & 3
    y_nrem = y[nrem_idx]
    nrem_minutes = len(y_nrem) / samples_per_minute
    rate_nrem = count_detections(y_nrem, merge_samples) / nrem_minutes

    # --- 3. First X minutes of NREM ---
    first_hour_samples = int(minutes * samples_per_minute)  # Samples in X minutes
    # check if I have enough samples
    if len(nrem_idx[0]) < first_hour_samples:
        less_than_hour = (first_hour_samples - len(nrem_idx[0])) / samples_per_minute
    else:
        less_than_hour = 0
    first_hour_nrem_idx = nrem_idx[0][:first_hour_samples]  # Get first hour of NREM
    y_nrem_1h = y[first_hour_nrem_idx]
    if minutes == 60:
        rate_nrem_1h = count_detections(y_nrem_1h, merge_samples) / (60 - less_than_hour)  # Normalize to per-minute rate
        nrem_1h_density = count_density(y_nrem_1h, merge_samples)
    else:
        # If not enough samples, calculate the rate based on available samples
        rate_nrem_1h = count_detections(y_nrem_1h, merge_samples) / minutes
        nrem_1h_density = count_density(y_nrem_1h, merge_samples)

    return rate_sleep, rate_nrem, rate_nrem_1h, less_than_hour, nrem_1h_density


def match_y_with_hypno_all(subjects):
    rates_sleep = {}
    rates_nrem = {}
    rates_first_nrem = {}
    for subj in subjects:
        if subj == '54':
            pkl_path = 'y_for_hypno_54.pkl'
            fif_path = r"C:\clean_zeeg\P54_mtl_clean_NEW.fif"
        elif subj == '49':
            pkl_path = 'y_for_hypno_49.pkl'
            fif_path = rf"C:\clean_zeeg\P{subj}_mtl_clean.fif"
        else:
            pkl_path = r"y_for_hypno.pkl"
            fif_path = rf"C:\clean_zeeg\P{subj}_mtl_clean.fif"
        if subj == '55':
            hypno_path = r"D:\Ichilov_scoring\P55_uncropped.txt"
        else:
            hypno_path = rf"D:\Ichilov_scoring\P{subj}.txt"

        y, hypno = match_y_with_hypno(fif_path, hypno_path, pkl_path, subj)
        rate_sleep, rate_nrem, rate_nrem_1h = calculate_rates_restricted(y, hypno, merge_window_sec=2)
        # rate_sleep, rate_nrem, rate_nrem_1h = calculate_rates(y, hypno)
        rates_sleep[id_name_map[subj]] = rate_sleep
        rates_nrem[id_name_map[subj]] = rate_nrem
        rates_first_nrem[id_name_map[subj]] = rate_nrem_1h

    return rates_sleep, rates_nrem, rates_first_nrem


def first_hour_spm(merge_window_sec=2):
    full_file = ['44', '46']
    non_full_file = ['46', '47', '48', '49', '51', '53', '54', '55', '56', '57', '58', '60']  # 45?
    first_hour_map = {'44': 'AO0', '46': 'MM1', '47': 'SS1', '48': 'TA1', '49': 'MF7', '51': 'RS4',
                      '53': 'IA2', '54': 'MA5', '55': 'NG6', '56': 'DR1', '57': 'MR9', '58': 'MF32'} # '42': 'DH3'
    # '013': '013', '017': '017', '018': '018', '025': '025',
    rates_sleep = {}
    rates_nrem = {}
    rates_first_nrem = {}
    dist_first_nrem = {}

    for id, name in first_hour_map.items():
        if id in full_file:
            pkl_path = r"y_for_hypno.pkl"
            fif_path = rf"C:\clean_zeeg\P{id}_mtl_clean.fif" if id != '54' else r"C:\clean_zeeg\P54_mtl_clean_NEW.fif"
            hypno_path = rf"D:\Ichilov_scoring\P{id}.txt"
        else:
            fif_path = rf"Z:\25. Interictal activities - Rotem\TLV\first_hour\clean\{name}_mtl_clean.fif"
            hypno_path = rf"Z:\40. Vlad\Spike\first_hour_hypno\{name}_first.txt"
            pkl_path = r'C:\repos\spikes_notebooks\paper\y_first_hour.pkl'

        y, hypno = match_y_with_hypno(fif_path, hypno_path, pkl_path, name)
        rate_sleep, rate_nrem, rate_nrem_1h, missing_minutes, avg_dist1 = calculate_rates_restricted(y, hypno, merge_window_sec=merge_window_sec)
        if missing_minutes > 0 and id not in ['42']:
            pkl_path = r"y_for_hypno.pkl" if id != '54' else 'y_for_hypno_54.pkl'
            fif_path = rf"C:\clean_zeeg\P{id}_mtl_clean.fif" if id != '54' else r"C:\clean_zeeg\P54_mtl_clean_NEW.fif"
            hypno_path = rf"D:\Ichilov_scoring\P{id}.txt" if id != '55' else r"D:\Ichilov_scoring\P55_uncropped.txt"
            y, hypno = match_y_with_hypno(fif_path, hypno_path, pkl_path, name)
            rate_sleep2, rate_nrem2, rate_nrem_1h2, missing_minutes2, avg_dist2 = calculate_rates_restricted(y, hypno,
                                                                                              merge_window_sec=merge_window_sec,
                                                                                              minutes=missing_minutes)
        # rate_sleep, rate_nrem, rate_nrem_1h = calculate_rates(y, hypno)
        rates_sleep[name] = rate_sleep
        rates_nrem[name] = rate_nrem
        if missing_minutes > 0 and id not in ['42']:
            rates_first_nrem[name] = ((rate_nrem_1h * (60 - missing_minutes)) + (rate_nrem_1h2 * missing_minutes)) / 60
            dist_first_nrem[name] = ((avg_dist1 * (60 - missing_minutes)) + (avg_dist2 * missing_minutes)) / 60
        else:
            rates_first_nrem[name] = rate_nrem_1h
            dist_first_nrem[name] = avg_dist1

    return rates_sleep, rates_nrem, rates_first_nrem, dist_first_nrem

# rate1, rate2, rate3 = match_y_with_hypno_all([x for x in list(id_name_map.keys()) if x not in ['49']])
# print("all sleep rates:", rate1)
# print("nrem rates:", rate2)
# print("first hour nrem rates:", rate3)

# calc_real_spm()
full_file = ['44']
non_full_file = ['42', '46', '47', '48', '49', '51', '53', '54', '55', '56', '57', '58', '60']  # 45?
first_hour_map = {'42': 'DH3', '44': 'AO0', '46': 'MM1', '47': 'SS1', '48': 'TA1', '49': 'MF7', '51': 'RS4',
                  '53': 'IA2', '54': 'MA5',
                  '55': 'NG6', '56': 'DR1', '57': 'MR9', '58': 'MF32', '60': 'SR34'}

# create first hour y_pkl
# get_depth_pred_unbalanced([v for k, v in first_hour_map.items() if k in non_full_file], 'y_first_hour.pkl',
#                           threshold=0.8, min_channels=2,
#                           raw_file=r"Z:\25. Interictal activities - Rotem\TLV\first_hour\clean\%s_mtl_clean.fif")


# updates nrem first hour rates:
# create rates according to hypnogram
# rates_sleep, rates_nrem, rates_first_nrem, avg_dist = first_hour_spm(merge_window_sec=10)
# print("all sleep rates:", rates_sleep)
# print("nrem rates:", rates_nrem)
# print("first hour nrem rates:", rates_first_nrem)
# for subj, rate in rates_first_nrem.items():
#     print(f"  {subj}: {rate:.4f}")
#
# print("average distance between detections in first hour of NREM:", avg_dist)
# for subj, dist in avg_dist.items():
#     print(f"  {subj}: {dist:.4f}")
#
# # avg of all subjects if not 0
# avg_sleep = np.mean([rate for rate in avg_dist.values() if rate > 0])
# print("Average distance between detections in first hour of NREM across all subjects:", avg_sleep)





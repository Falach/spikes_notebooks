"""
Create Detection Validation EDF Files (v2)
==========================================
For each patient, find 5-min clean epochs with 15-30 detected epileptic spikes,
create EDF files with raw + bipolar depth channels AND companion EOG-only EDFs.
Assigns randomized anonymous IDs per subject.

Usage:
    python create_detection_edfs_v2.py
"""

import numpy as np
import pandas as pd
import mne
import joblib
import os
import time
import random
import warnings
import antropy as ant
import scipy.stats as sp_stats
import mne_features.univariate
import pyedflib

warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETERS
# ============================================================================
sr = 1000
window_size = 250  # samples (250 ms at 1 kHz)
threshold = 0.8
min_spikes = 15
max_spikes = 30
epoch_duration = 300  # seconds (5 minutes)
min_clean_pct = 0.95
min_gap_between_files = 15 * 60  # 15 minutes in seconds
min_spike_gap = 1.0  # seconds between unique spike events
max_files_per_subject = 2
edf_phys_range = 3200  # fixed ±3200 µV for all channels

# Paths
mtl_path = r'D:\clean_zeeg\P%s_mtl_clean.fif'
model_path = r'C:\repos\depth_ieds\paper\model_V6_90_10.pkl'
output_dir = r'c:\repos\spikes_notebooks\detection_validation_edfs_v2'

# Subject lists
tlv_subjects = ['013', '017', '018', '025', '38', '39', '44', '46', '47',
                '48', '49', '51', '53', '54', '55', '56', '57']
bonn_subjects = ['707', '708', '709', '710', '711', '712', '713', '714',
                 '715', '723', '724', '728', '731', '733', '734', '735',
                 '737', '744', '746', '752']
milan_subjects = ['801', '802', '804', '805', '807', '809', '810', '812',
                  '813', '814', '815', '816', '817', '818']
all_subjects = tlv_subjects + bonn_subjects + milan_subjects

# Depth channel list (corrected comma after RA3)
depth_channels = [
    'RAH1', 'LAH1', 'RA1', 'LA1', 'LEC1', 'REC1', 'RPHG1', 'LPHG1',
    'RMH1', 'LMH1', 'LH1', 'RH1', 'RA3',
    'RAH2', 'LAH2', 'RA2', 'LA2', 'LEC2', 'REC2', 'RPHG2', 'LPHG2',
    'RMH2', 'LMH2', 'LH2', 'RH2'
]

# EOG channels to check for
eog_channel_candidates = ['EOG', 'EOG1', 'EOG2']


# ============================================================================
# ID GENERATION
# ============================================================================
def generate_random_ids(n_subjects, seed=42):
    """Generate unique random 2-digit IDs for temporal and zeeg files."""
    rng = random.Random(seed)
    pool = list(range(1, 100))  # 01-99

    rng.shuffle(pool)
    temporal_ids = pool[:n_subjects]

    rng.shuffle(pool)
    zeeg_ids = pool[:n_subjects]

    return temporal_ids, zeeg_ids


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_epochs_top_features(epochs, subj, sfreq):
    """Extract the top features used by the depth spike detection model."""
    epochs = np.array(epochs)
    mobility, complexity = ant.hjorth_params(epochs, axis=1)
    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'kurtosis': sp_stats.kurtosis(epochs, axis=1),
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity,
        'ptp_amp': np.ptp(epochs, axis=1),
        'samp_entropy': np.apply_along_axis(
            ant.sample_entropy, axis=1, arr=epochs
        ),
    }
    kaiser = mne_features.univariate.compute_teager_kaiser_energy(epochs)
    reshaped = np.array(kaiser).reshape(-1, 12)
    X_new = pd.DataFrame(reshaped, columns=[
        f'teager_kaiser_energy_{i}_mean' if j % 2 == 0
        else f'teager_kaiser_energy_{i}_std'
        for i in range(6) for j in range(2)
    ])
    feat = pd.DataFrame(feat)
    feat = pd.concat([feat, X_new], axis=1)
    return feat


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def detect_spikes_channel(raw, chan, subj, depth_model, feature_names):
    """Run spike detection on a single channel. Returns spike times (sec)."""
    if chan not in raw.ch_names:
        return []

    chan_data = raw.copy().pick([chan]).get_data(
        reject_by_annotation='NaN'
    ).flatten()
    chan_norm = (chan_data - np.nanmean(chan_data)) / np.nanstd(chan_data)

    epochs, epoch_positions = [], []
    for i in range(0, len(chan_norm) - 4 * window_size, window_size):
        if not np.isnan(chan_norm[i:i + window_size]).any():
            epochs.append(chan_norm[i:i + window_size])
            epoch_positions.append(i)

    if len(epochs) == 0:
        return []

    feat = extract_epochs_top_features(epochs, subj, raw.info['sfreq'])
    valid_norm = chan_norm[~np.isnan(chan_norm)]
    feat['chan_name'] = chan
    feat['chan_ptp'] = np.ptp(valid_norm)
    feat['chan_skew'] = sp_stats.skew(valid_norm)
    feat['chan_kurt'] = sp_stats.kurtosis(valid_norm)

    predictions = depth_model.predict_proba(feat[feature_names])
    detected_mask = predictions[:, 1] >= threshold

    return [epoch_positions[j] / raw.info['sfreq']
            for j in range(len(epochs)) if detected_mask[j]]


def get_annotation_coverage(raw, t_start, t_end):
    """Fraction of [t_start, t_end] covered by annotations."""
    total = 0.0
    for annot in raw.annotations:
        a_s = annot['onset'] - raw.first_time
        a_e = a_s + annot['duration']
        ov_s, ov_e = max(a_s, t_start), min(a_e, t_end)
        if ov_s < ov_e:
            total += ov_e - ov_s
    return total / (t_end - t_start)


def count_unique_spikes_in_window(all_spike_times, t_start, t_end, min_gap):
    """Count unique spike events in window, ≥min_gap apart."""
    times = sorted(t for t, _ in all_spike_times if t_start <= t < t_end)
    if not times:
        return 0
    unique = [times[0]]
    for t in times[1:]:
        if t - unique[-1] >= min_gap:
            unique.append(t)
    return len(unique)


def get_spike_annotations_in_window(channel_detections, t_start, t_end):
    """Get per-channel spike annotations within window as (rel_time, chan)."""
    annots = []
    for chan, times in channel_detections.items():
        for t in times:
            if t_start <= t < t_end:
                annots.append((t - t_start, chan))
    annots.sort(key=lambda x: x[0])
    return annots


def find_bipolar_partner(chan, raw_ch_names):
    """Try N+1, then N+2. Returns partner name or None."""
    prefix = chan.rstrip('0123456789')
    num_str = chan[len(prefix):]
    if not num_str:
        return None
    num = int(num_str)
    for offset in [1, 2]:
        candidate = f'{prefix}{num + offset}'
        if candidate in raw_ch_names:
            return candidate
    return None


def find_valid_windows(raw, all_spike_times, step=10):
    """Find 5-min windows with min_spikes ≤ spikes ≤ max_spikes, ≥95% clean."""
    valid = []
    for t_start in np.arange(0, raw.times[-1] - epoch_duration, step):
        t_end = t_start + epoch_duration
        clean_pct = 1.0 - get_annotation_coverage(raw, t_start, t_end)
        if clean_pct < min_clean_pct:
            continue
        n = count_unique_spikes_in_window(
            all_spike_times, t_start, t_end, min_spike_gap)
        if min_spikes <= n <= max_spikes:
            valid.append((t_start, n, clean_pct))
    return valid


def select_windows(valid_windows, min_gap):
    """Select up to max_files_per_subject windows, ≥min_gap apart."""
    if not valid_windows:
        return []
    # Prefer windows with more spikes (descending), then earlier time
    sw = sorted(valid_windows, key=lambda x: (-x[1], x[0]))
    selected = [sw[0]]
    for w in sw[1:]:
        if len(selected) >= max_files_per_subject:
            break
        if all(abs(w[0] - s[0]) >= min_gap for s in selected):
            selected.append(w)
    selected.sort(key=lambda x: x[0])
    return selected


def save_edf(fpath, data_uv, ch_names, sfreq, patient_name, annotations=None):
    """
    Write EDF+ with fixed ±3200 µV physical range.
    data_uv: ndarray in µV units, shape (n_channels, n_samples).
    """
    n_ch = len(ch_names)
    f = pyedflib.EdfWriter(fpath, n_ch, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        f.setPatientName(patient_name)
        for i in range(n_ch):
            f.setLabel(i, ch_names[i])
            f.setPhysicalDimension(i, 'uV')
            f.setSamplefrequency(i, sfreq)
            f.setPhysicalMaximum(i, edf_phys_range)
            f.setPhysicalMinimum(i, -edf_phys_range)
            f.setDigitalMaximum(i, 32767)
            f.setDigitalMinimum(i, -32768)
        if annotations:
            for onset, duration, desc in annotations:
                f.writeAnnotation(onset, duration, desc)
        clipped = np.clip(data_uv, -edf_phys_range, edf_phys_range)
        f.writeSamples(list(clipped))
    finally:
        f.close()


def prepare_temporal_data(raw_cropped, available_channels, bipolar_pairs):
    """Prepare sorted raw + sorted bipolar channel data in µV."""
    sfreq = raw_cropped.info['sfreq']

    # Raw channels (already sorted)
    raw_pick = raw_cropped.copy().pick(available_channels)
    raw_data = raw_pick.get_data() * 1e6  # V -> µV
    raw_names = list(raw_pick.ch_names)

    # Bipolar channels (already sorted)
    bp_data, bp_names = [], []
    for ch1, ch2 in bipolar_pairs:
        d1 = raw_cropped.copy().pick([ch1]).get_data()
        d2 = raw_cropped.copy().pick([ch2]).get_data()
        bp_data.append((d1 - d2) * 1e6)
        bp_names.append(f'{ch1}-{ch2}')

    if bp_data:
        all_data = np.vstack([raw_data] + bp_data)
    else:
        all_data = raw_data
    all_names = raw_names + bp_names
    return all_data, all_names, sfreq


def prepare_eog_data(raw_cropped, available_eog):
    """Prepare EOG channel data in µV."""
    sfreq = raw_cropped.info['sfreq']
    eog_pick = raw_cropped.copy().pick(available_eog)
    eog_data = eog_pick.get_data() * 1e6  # V -> µV
    return eog_data, list(eog_pick.ch_names), sfreq


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model = joblib.load(model_path)
    depth_model, feature_names = model['model'], model['features']

    # Generate random anonymous IDs
    n = len(all_subjects)
    temporal_ids, zeeg_ids = generate_random_ids(n)
    subj_id_map = {
        subj: {'temporal_id': f'{temporal_ids[i]:02d}',
               'zeeg_id': f'{zeeg_ids[i]:02d}'}
        for i, subj in enumerate(all_subjects)
    }

    summary_rows = []

    for subj in all_subjects:
        print(f"\n{'=' * 70}")
        print(f"  Processing subject P{subj}")
        print(f"{'=' * 70}")

        # --- Load raw ---
        if subj == '54':
            fif_path = r"D:\clean_zeeg\P54_mtl_clean_NEW.fif"
        else:
            fif_path = mtl_path % subj
        if not os.path.exists(fif_path):
            print(f"  WARNING: File not found: {fif_path}, skipping.")
            continue

        raw = mne.io.read_raw(fif_path, preload=True, verbose=False)
        print(f"  Duration: {raw.times[-1]:.0f}s ({raw.times[-1]/60:.1f}min)")

        # --- Find available depth channels (sorted) ---
        available_channels = sorted(
            ch for ch in depth_channels if ch in raw.ch_names
        )
        if not available_channels:
            print(f"  WARNING: No depth channels found, skipping.")
            continue
        print(f"  Depth channels: {available_channels}")

        # --- Find available EOG channels (sorted) ---
        available_eog = sorted(
            ch for ch in eog_channel_candidates if ch in raw.ch_names
        )
        if not available_eog:
            print(f"  WARNING: No EOG channels found for P{subj}.")
        else:
            print(f"  EOG channels: {available_eog}")

        # --- Build bipolar pairs (sorted) ---
        # Partner channels can be outside the depth_channels list
        # (e.g. RAH3 for RAH2) as long as they exist in the raw file
        bipolar_pairs = []
        for chan in available_channels:
            partner = find_bipolar_partner(chan, raw.ch_names)
            if partner is not None:
                bipolar_pairs.append((chan, partner))
                if partner not in depth_channels:
                    print(f"  INFO: Using {partner} (not in depth list) "
                          f"as bipolar ref for {chan}")
            else:
                print(f"  NOTE: No bipolar partner for {chan} "
                      f"in raw file (tried +1 and +2), skipping.")
        bipolar_pairs.sort(key=lambda x: x[0])
        print(f"  Bipolar: {[f'{a}-{b}' for a, b in bipolar_pairs]}")

        # --- Spike detection per channel ---
        print(f"  Running spike detection...")
        channel_detections = {}
        all_spike_times = []
        for chan in available_channels:
            t0 = time.time()
            stimes = detect_spikes_channel(
                raw, chan, subj, depth_model, feature_names)
            channel_detections[chan] = stimes
            for t in stimes:
                all_spike_times.append((t, chan))
            print(f"    {chan}: {len(stimes)} detections "
                  f"({time.time()-t0:.1f}s)")

        all_spike_times.sort(key=lambda x: x[0])
        print(f"  Total detections: {len(all_spike_times)}")

        if len(all_spike_times) == 0:
            print(f"  WARNING: No spikes detected, skipping.")
            continue

        # --- Find valid 5-min windows (15-30 spikes) ---
        print(f"  Searching for valid windows (15-30 spikes)...")
        valid_windows = find_valid_windows(raw, all_spike_times, step=10)
        print(f"  Found {len(valid_windows)} valid windows")
        if valid_windows:
            from collections import Counter
            spike_dist = Counter(w[1] for w in valid_windows)
            print(f"  Spike count distribution: "
                  f"{dict(sorted(spike_dist.items()))}")

        # Relax if needed
        used_relaxed = False
        if not valid_windows:
            print(f"  WARNING: No window meets strict criteria. Relaxing...")
            for relax_pct in [0.80, 0.60, 0.40, 0.0]:
                for t_s in np.arange(0, raw.times[-1] - epoch_duration, 10):
                    cp = 1.0 - get_annotation_coverage(
                        raw, t_s, t_s + epoch_duration)
                    if cp < relax_pct:
                        continue
                    ns = count_unique_spikes_in_window(
                        all_spike_times, t_s, t_s + epoch_duration,
                        min_spike_gap)
                    if ns >= min_spikes:
                        valid_windows.append((t_s, ns, cp))
                if valid_windows:
                    used_relaxed = True
                    print(f"  Found {len(valid_windows)} windows at "
                          f"≥{relax_pct*100:.0f}% clean")
                    break

        if not valid_windows:
            # Last resort: best available
            print(f"  WARNING P{subj}: Taking best available window.")
            best_s, best_n, best_c = 0, 0, 0
            for t_s in np.arange(0, raw.times[-1] - epoch_duration, 10):
                af = get_annotation_coverage(
                    raw, t_s, t_s + epoch_duration)
                ns = count_unique_spikes_in_window(
                    all_spike_times, t_s, t_s + epoch_duration, min_spike_gap)
                if ns > best_n:
                    best_n, best_s, best_c = ns, t_s, 1.0 - af
            if best_n > 0:
                valid_windows = [(best_s, best_n, best_c)]
                used_relaxed = True
                print(f"  Best: {best_n} spikes, {best_c*100:.1f}% clean")

        if not valid_windows:
            print(f"  WARNING: No usable windows, skipping.")
            continue

        # --- Select up to 2 windows ---
        selected = select_windows(valid_windows, min_gap_between_files)
        print(f"  Selected {len(selected)} window(s):")
        for i, (ts, ns, cp) in enumerate(selected):
            # Verification re-count
            verify_n = count_unique_spikes_in_window(
                all_spike_times, ts, ts + epoch_duration, min_spike_gap)
            print(f"    File {i+1}: start={ts:.0f}s ({ts/60:.1f}min), "
                  f"spikes={ns} (verified={verify_n}), clean={cp*100:.1f}%"
                  f"{' (RELAXED)' if used_relaxed else ''}")

        # --- Get IDs ---
        tid = subj_id_map[subj]['temporal_id']
        zid = subj_id_map[subj]['zeeg_id']

        # --- Create EDF files ---
        for fi, (t_start, n_spikes, clean_pct) in enumerate(selected):
            t_end = t_start + epoch_duration
            raw_cropped = raw.copy().crop(tmin=t_start, tmax=t_end)
            sfx = f"_{fi+1}" if len(selected) > 1 else ""

            # -- Spike annotations for this window --
            spike_annots = get_spike_annotations_in_window(
                channel_detections, t_start, t_end)
            edf_annots = [(a[0], 0.25, a[1]) for a in spike_annots]

            # -- TEMPORAL EDF --
            t_name = f"Falach_temporal_{tid}{sfx}"
            t_data, t_chs, sfreq = prepare_temporal_data(
                raw_cropped, available_channels, bipolar_pairs)
            t_fname = f"{t_name}.edf"
            t_fpath = os.path.join(output_dir, t_fname)
            save_edf(t_fpath, t_data, t_chs, sfreq, t_name, edf_annots)
            print(f"  Saved: {t_fname}")

            # -- ZEEG (EOG) EDF --
            z_fname, z_fpath = '', ''
            if available_eog:
                z_name = f"Falach_zeeg_{zid}{sfx}"
                z_data, z_chs, sfreq = prepare_eog_data(
                    raw_cropped, available_eog)
                z_fname = f"{z_name}.edf"
                z_fpath = os.path.join(output_dir, z_fname)
                save_edf(z_fpath, z_data, z_chs, sfreq, z_name)
                print(f"  Saved: {z_fname}")
            else:
                print(f"  Skipped EOG EDF (no EOG channels).")

            # -- Summary row --
            bp_names = [f'{a}-{b}' for a, b in bipolar_pairs]
            summary_rows.append({
                'subject': subj,
                'file_index': fi + 1,
                'temporal_id': tid,
                'zeeg_id': zid,
                'temporal_file': t_fname,
                'zeeg_file': z_fname,
                'temporal_header_name': f"Falach_temporal_{tid}{sfx}",
                'zeeg_header_name': f"Falach_zeeg_{zid}{sfx}" if z_fname else '',
                'start_time_sec': t_start,
                'start_time_min': round(t_start / 60, 2),
                'duration_sec': epoch_duration,
                'n_unique_spikes': n_spikes,
                'n_total_annotations': len(spike_annots),
                'pct_clean': round(clean_pct * 100, 2),
                'relaxed_criteria': used_relaxed,
                'channels_raw': ', '.join(available_channels),
                'channels_bipolar': ', '.join(bp_names),
                'channels_eog': ', '.join(available_eog),
                'n_raw_channels': len(available_channels),
                'n_bipolar_channels': len(bipolar_pairs),
                'n_eog_channels': len(available_eog),
            })

        del raw  # free memory

    # --- Save summary ---
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        spath = os.path.join(output_dir, 'summary.csv')
        df.to_csv(spath, index=False)
        print(f"\n{'=' * 70}")
        print(f"Summary saved to: {spath}")
        print(f"Total files created: {len(summary_rows)} temporal + "
              f"{sum(1 for r in summary_rows if r['zeeg_file'])} zeeg")
        print(df[['subject', 'temporal_id', 'zeeg_id', 'file_index',
                   'start_time_min', 'n_unique_spikes', 'pct_clean'
                   ]].to_string())
    else:
        print("\nNo EDF files were created.")


if __name__ == '__main__':
    main()

"""
Extract chosen temporal EDF files without annotations.
Reads existing files from detection_validation_edfs_v2 and re-saves them
to a 'chosen' subfolder with identical data but no annotations.
"""

import os
import numpy as np
import pyedflib

# Source and output directories
source_dir = r'c:\repos\spikes_notebooks\detection_validation_edfs_v2'
output_dir = r'C:\repos\spikes_notebooks\detection_validation_edfs_v2\chosen'

# Chosen files: temporal_id + suffix
chosen_files = [
    '73_2', '34_2', '02_1', '16_2', '63_2',
    '11_2', '57_1', '46_1', '69_1', '94_2',
    '48_2', '51_1', '37_1', '24_2', '40_1',
]

EDF_PHYS_RANGE = 3200  # same fixed range as original

def copy_edf_no_annotations(src_path, dst_path):
    """Read an EDF file and re-write it without any annotations."""
    reader = pyedflib.EdfReader(src_path)
    try:
        n_channels = reader.signals_in_file
        patient_name = reader.getPatientName()
        ch_labels = [reader.getLabel(i) for i in range(n_channels)]
        sfreqs = [reader.getSampleFrequency(i) for i in range(n_channels)]
        phys_dims = [reader.getPhysicalDimension(i) for i in range(n_channels)]

        # Read all channel data
        data = []
        for i in range(n_channels):
            data.append(reader.readSignal(i))
    finally:
        reader.close()

    # Write without annotations
    writer = pyedflib.EdfWriter(dst_path, n_channels,
                                file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setPatientName(patient_name)
        for i in range(n_channels):
            writer.setLabel(i, ch_labels[i])
            writer.setPhysicalDimension(i, phys_dims[i])
            writer.setSamplefrequency(i, sfreqs[i])
            writer.setPhysicalMaximum(i, EDF_PHYS_RANGE)
            writer.setPhysicalMinimum(i, -EDF_PHYS_RANGE)
            writer.setDigitalMaximum(i, 32767)
            writer.setDigitalMinimum(i, -32768)
        # No annotations written
        writer.writeSamples(data)
    finally:
        writer.close()


def main():
    os.makedirs(output_dir, exist_ok=True)

    for fid in chosen_files:
        fname = f"Falach_temporal_{fid}.edf"
        src = os.path.join(source_dir, fname)

        if not os.path.exists(src):
            print(f"WARNING: {fname} not found, skipping.")
            continue

        dst = os.path.join(output_dir, fname)
        copy_edf_no_annotations(src, dst)
        print(f"  Saved: {fname}")

    print(f"\nDone. {len(chosen_files)} files saved to {output_dir}")


if __name__ == '__main__':
    main()

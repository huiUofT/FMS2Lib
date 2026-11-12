#!/usr/bin/env python3
"""
Clean an MGF file:
1. Drop fragment peaks below a relative-intensity threshold (relative to the
   spectrum maximum; default 0.01 = 1%).
2. Drop isotope fragment peaks (M+1, M+2, and so on).
"""
import re
from pathlib import Path
from tqdm import tqdm

def is_isotope_peak(mz, intensity, peaks, tolerance=0.01):
    """
    Decide whether a peak is an isotope peak.

    Args:
        mz: m/z of the current peak
        intensity: intensity of the current peak
        peaks: all peaks as [(mz, intensity), ...]
        tolerance: m/z tolerance

    Returns:
        bool: True if the peak is an isotope peak
    """
    # M+1: 13C, m/z difference about 1.003
    # M+2: m/z difference about 2.006
    # M+3: m/z difference about 3.009

    for other_mz, other_intensity in peaks:
        if other_intensity <= intensity:  # only compare against stronger peaks
            continue

        mz_diff = mz - other_mz

        # M+1: 13C isotope, m/z difference about 1.00335
        if abs(mz_diff - 1.00335) < tolerance:
            if intensity < other_intensity * 0.5:
                return True

        # M+2: m/z difference about 2.0067 (13C2 or 18O)
        if abs(mz_diff - 2.0067) < tolerance:
            if intensity < other_intensity * 0.3:
                return True

        # M+3: m/z difference about 3.01005
        if abs(mz_diff - 3.01005) < tolerance:
            if intensity < other_intensity * 0.2:
                return True

        # Other isotope offsets (for example M+0.5, M+1.5)
        for iso_diff in [0.5, 1.5, 2.5]:
            if abs(mz_diff - iso_diff) < tolerance:
                if intensity < other_intensity * 0.4:
                    return True

    return False

def clean_mgf_file(input_file, output_file, intensity_threshold=0.01, remove_isotopes=True):
    """
    Clean an MGF file.

    Args:
        input_file: input MGF path
        output_file: output MGF path
        intensity_threshold: relative intensity cutoff versus the spectrum maximum
        remove_isotopes: whether to drop isotope peaks
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Relative intensity threshold: {intensity_threshold} (relative to the spectrum maximum)")
    print(f"Remove isotope peaks: {remove_isotopes}")

    stats = {
        'total_spectra': 0,
        'total_peaks_before': 0,
        'total_peaks_after': 0,
        'removed_low_intensity': 0,
        'removed_isotopes': 0
    }

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        metadata_lines = []
        peaks = []
        in_spectrum = False

        print("Processing file...")

        for line in tqdm(infile, desc="Processing"):
            line_stripped = line.rstrip()

            if line_stripped.startswith('BEGIN IONS'):
                if in_spectrum:
                    _write_spectrum(metadata_lines, peaks, outfile,
                                  intensity_threshold, remove_isotopes, stats)

                in_spectrum = True
                metadata_lines = [line_stripped]
                peaks = []
                stats['total_spectra'] += 1
                continue

            if line_stripped.startswith('END IONS'):
                if in_spectrum:
                    _write_spectrum(metadata_lines, peaks, outfile,
                                  intensity_threshold, remove_isotopes, stats)
                    in_spectrum = False
                continue

            if in_spectrum:
                match = re.match(r'^(\d+\.?\d*)\s+(\d+\.?\d*)', line_stripped)
                if match:
                    mz = float(match.group(1))
                    intensity = float(match.group(2))
                    peaks.append((mz, intensity))
                else:
                    metadata_lines.append(line_stripped)

        # Last spectrum if the file does not end with END IONS
        if in_spectrum and peaks:
            _write_spectrum(metadata_lines, peaks, outfile,
                          intensity_threshold, remove_isotopes, stats)

    print("\n" + "="*80)
    print("Cleaning finished")
    print("="*80)
    print(f"Spectra: {stats['total_spectra']}")
    print(f"Peaks before: {stats['total_peaks_before']}")
    print(f"Peaks after: {stats['total_peaks_after']}")
    print(f"Removed low-intensity peaks (<{intensity_threshold} * max): {stats['removed_low_intensity']}")
    print(f"Removed isotope peaks: {stats['removed_isotopes']}")
    if stats['total_spectra'] > 0:
        avg_before = stats['total_peaks_before'] / stats['total_spectra']
        avg_after = stats['total_peaks_after'] / stats['total_spectra']
        print(f"Mean peaks per spectrum: {avg_before:.2f} -> {avg_after:.2f}")
    print("="*80)

def _write_spectrum(metadata_lines, peaks, outfile, intensity_threshold, remove_isotopes, stats):
    """Write one cleaned spectrum."""
    stats['total_peaks_before'] += len(peaks)

    # Relative-intensity filter against this spectrum's maximum peak
    filtered = []
    if peaks:
        max_intensity = max(intensity for _, intensity in peaks)
        if max_intensity > 0:
            abs_cutoff = intensity_threshold * max_intensity
            for mz, intensity in peaks:
                if intensity < abs_cutoff:
                    stats['removed_low_intensity'] += 1
                    continue
                filtered.append((mz, intensity))
        else:
            filtered = list(peaks)

    if remove_isotopes and filtered:
        filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)

        final_peaks = []
        for mz, intensity in filtered_sorted:
            if is_isotope_peak(mz, intensity, final_peaks):
                stats['removed_isotopes'] += 1
                continue
            final_peaks.append((mz, intensity))

        filtered = final_peaks

    stats['total_peaks_after'] += len(filtered)

    for meta_line in metadata_lines:
        outfile.write(meta_line + '\n')

    filtered.sort(key=lambda x: x[0])
    for mz, intensity in filtered:
        outfile.write(f"{mz} {intensity}\n")

    outfile.write('END IONS\n')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Clean an MGF file: drop low relative-intensity peaks and isotope peaks')
    parser.add_argument('--input', type=str, required=True,
                       help='Input MGF path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output MGF path (default: <input>_cleaned.mgf)')
    parser.add_argument('--intensity-threshold', type=float, default=0.01,
                       help='Relative intensity threshold versus the spectrum maximum. Default 0.01 = 1%%')
    parser.add_argument('--keep-isotopes', action='store_true',
                       help='Keep isotope peaks (default: remove them)')

    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_cleaned.mgf")

    clean_mgf_file(
        args.input,
        args.output,
        intensity_threshold=args.intensity_threshold,
        remove_isotopes=not args.keep_isotopes
    )

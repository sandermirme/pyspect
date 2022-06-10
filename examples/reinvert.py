import bisect
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent / "../src"))

import pyspect.inversion
import pyspect.datafiles

inverter = pyspect.inversion.load_inverter(
    open("inverters/v14.1-hrnd-elm25-chv/ions-pos-v14.1-hrnd-elm25-chv.inverter"))

data = pyspect.datafiles.RecordsFiles()
data.load("data/20220524-block.records")

spectra_data = pyspect.datafiles.RecordsFiles()
spectra_data.load("data/20220524-block-ions.spectra")

for s in spectra_data.spectra:
    if s.name == "ions pos":
        spectrum = s
        break
else:
    raise RuntimeError("Spectrum not found")

em_range_begin, em_range_end = data.electrometer_groups["pos"]

for i in range(data.count()):
    if data.opmode[i] != "ions":
        continue

    current = data.current[i][em_range_begin:em_range_end + 1]
    variance = data.current_variance[i][em_range_begin:em_range_end + 1]

    reinv_value, reinv_cov = inverter.invert(current, variance)

    spectrum_index = bisect.bisect_left(spectrum.begin_time, data.begin_time[i])
    if spectrum_index != len(spectrum.begin_time) and spectrum.begin_time[spectrum_index] == data.begin_time[i]:
        orig_spect = np.asarray(spectrum.value[spectrum_index])
        for j in range(len(orig_spect)):
            print(f"{j:3} {orig_spect[j]:6.2f} {reinv_value[j]:6.2f}")
        print(f"sum {orig_spect.sum():6.2f} {reinv_value.sum():6.2f}")

        input()
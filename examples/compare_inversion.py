import bisect
import math
import pathlib
import sys

import numpy as np

# Add library search path relative to the examples directory
sys.path.insert(0, str(pathlib.Path(__file__).parent / "../src"))

import pyspect.inversion
import pyspect.datafiles

inverter = pyspect.inversion.load_inverter(
    open("inverters/v14.1-hrnd-elm25-chv/ions-pos-v14.1-hrnd-elm25-chv.inverter"))

data = pyspect.datafiles.RecordsFiles()
data.load("dataorig/20220524-block.records")

spectra_data = pyspect.datafiles.RecordsFiles()
spectra_data.load("dataorig/20220524-block-ions.spectra")

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
    current_variance = data.current_variance[i][em_range_begin:em_range_end + 1]

    reinv_distribution, reinv_distribution_cov = inverter.invert(current, current_variance)

    spectrum_index = bisect.bisect_left(spectrum.begin_time, data.begin_time[i])
    if spectrum_index != len(spectrum.begin_time) and spectrum.begin_time[spectrum_index] == data.begin_time[i]:
        orig_distribution = np.asarray(spectrum.value[spectrum_index])

        sqdiff = math.sqrt(((orig_distribution - reinv_distribution) ** 2).mean())
        relsum = orig_distribution.sum() / reinv_distribution.sum()

        print(f"{i:5} {data.begin_time[i]} {sqdiff:9.6f} {relsum:9.6f}")

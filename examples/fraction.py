from airel import pyspect
from airel.pyspect.datafiles import SpectrumScale

import matplotlib.pyplot as plt

MILLIKAN = pyspect.millikan.Millikan()
NM = 1e-9
CM2VS = 1e-4

spectra_data = pyspect.datafiles.RecordsFiles()
spectra_data.load("data/20220524-block-ions.spectra")

d_limits = (5.0, 10.0)
z_limits = [MILLIKAN.dtok(d * NM) / CM2VS for d in d_limits]

limits = {
    SpectrumScale.DIAMETER: d_limits,
    SpectrumScale.MOBILITY: z_limits,
}

print(limits)

f, ax = plt.subplots()

for spectra in spectra_data.spectra:
    concentration = spectra.fraction_concentration(limits[spectra.scale])
    ax.scatter(spectra.begin_time, concentration, label=spectra.name, s=3)

ax.set_yscale("log")
ax.legend()
plt.show()

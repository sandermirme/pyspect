import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "../src"))

import pyspect.inversion
import pyspect.datafiles

inverters = {
    "ions": [
        ("pos", pyspect.inversion.load_inverter(
            open("inverters/v14.1-hrnd-elm25-chv/ions-pos-v14.1-hrnd-elm25-chv.inverter"))),
        ("neg", pyspect.inversion.load_inverter(
            open("inverters/v14.1-hrnd-elm25-chv/ions-neg-v14.1-hrnd-elm25-chv.inverter")))
    ],
    "particles": [
        ("pos", pyspect.inversion.load_inverter(
            open("inverters/v14.1-hrnd-elm25-chv/particles-pos-v14-hrnd-elm25-chv.inverter"))),
        ("neg", pyspect.inversion.load_inverter(
            open("inverters/v14.1-hrnd-elm25-chv/particles-neg-v14-hrnd-elm25-chv.inverter")))
    ]
}

data = pyspect.datafiles.RecordsFiles()
data.load("data/20220524-block.records")

for i in range(data.count()):
    opmode = data.opmode[i]
    if opmode not in inverters:
        continue

    begin_time = data.begin_time[i]
    end_time = data.end_time[i]

    for polarity, inverter in inverters[opmode]:
        em_range_begin, em_range_end = data.electrometer_groups[polarity]

        current = data.current[i][em_range_begin:em_range_end + 1]
        current_variance = data.current_variance[i][em_range_begin:em_range_end + 1]

        reinv_distribution, reinv_distribution_cov = inverter.invert(current, current_variance)

        # Do something with the result. Now just prints first 5 nd points
        print(f"{begin_time} {end_time} {opmode} {polarity} {reinv_distribution[:5]}")

import copy
from math import cos, pi, isnan, sqrt
from typing import Union

import numpy
import numpy as np
import scipy.linalg
import yaml

from .. import millikan


def fmt(a, f="5.2f"):
    fm = "{{:{}}}".format(f)
    return " ".join(fm.format(x) for x in a)


class InversionException(Exception):
    pass


_INVERTER_ATTRIBUTES = [
    ("initial_regul_coef", float),
    ("next_regul_coef", float),
    ("neg_remove_iterations", int),
    ("neg_remove_coef", float),
    ("smoothing_coef", float),
    ("regul_smoothing_coef", float),
    ("regul_off_diag_correction", bool),
    ("regul_limit", float),
]


class Inverter:
    def __init__(self):
        self.initial_regul_coef = 0.1
        self.next_regul_coef = 2.0
        self.neg_remove_iterations = 10
        self.neg_remove_coef = 0.5
        self.smoothing_coef = 0.0
        self.regul_smoothing_coef = 2.0
        self.regul_off_diag_correction = False
        self.regul_limit = 0
        self.instrument_matrix = None
        self.post_matrix = None
        self.smoothing_matrix = None
        self.regul_smoothing_matrix = None
        self.ignore_variance = False

        self._cached_smoothing_opts = None
        self._cached_regul_smoothing_opts = None

    def print_opts(
        self,
    ):
        names = [
            "neg_remove_coef",
            "neg_remove_iterations",
            "initial_regul_coef",
            "next_regul_coef",
            "regul_limit",
            "regul_off_diag_correction",
            "regul_smoothing_coef",
        ]

        for n in names:
            print("inv.{} = {}".format(n, repr(getattr(self, n))))

    def prepare(self):
        opts = (self.instrument_matrix.shape[0], self.smoothing_coef)
        if self._cached_smoothing_opts != opts:
            self.smoothing_matrix = make_smoothing_matrix(*opts)
            self._cached_smoothing_opts = opts

        opts = (self.instrument_matrix.shape[1], self.regul_smoothing_coef)
        if self._cached_regul_smoothing_opts != opts:
            self.regul_smoothing_matrix = make_smoothing_matrix(*opts)
            self._cached_regul_smoothing_opts = opts

    def subinvert(
        self,
        instrument_matrix: numpy.ndarray,
        signal: numpy.ndarray,
        variance: numpy.ndarray,
        vparand: Union[None, numpy.ndarray] = None,
    ):

        H = instrument_matrix

        # print "invsig", fmt(signal, "+8.4f")
        # print "invvar", fmt(variance, "+8.4f")

        inputs = H.shape[0]
        outputs = H.shape[1]

        if inputs != variance.size:
            raise InversionException(
                "Instrument matrix does not match variance vector size"
            )

        Dinv = np.zeros((inputs, inputs))

        if self.ignore_variance:
            for i in range(inputs):
                Dinv[i, i] = 1
        else:
            for i in range(inputs):
                v = variance[i]
                if v == 0:
                    raise InversionException("A variance is 0.")
                Dinv[i, i] = 1 / v

        HT = H.T

        W = np.dot(HT, Dinv)
        Vorig = np.dot(W, H)
        V = Vorig.copy()

        if inputs != len(signal):
            raise InversionException(
                "Instrument matrix does not match record vector size"
            )

        Wy = np.dot(W, signal)

        if vparand is None:
            for i in range(outputs):
                V[i, i] *= 1.0 + self.initial_regul_coef

            Vcainv = scipy.linalg.inv(V)

            V = Vorig.copy()

            if Vcainv.shape[0] == 0:
                raise InversionException("Unable to invert vca.")

            Fa = np.dot(Vcainv, Wy)

            vparand = np.empty(outputs)

            for i in range(outputs):
                x = self.next_regul_coef * Vcainv[i, i] / (pow(Fa[i], 2))
                if x < self.initial_regul_coef:
                    x = self.initial_regul_coef

                if self.regul_limit is not None:
                    if Fa[i] < 0.0:
                        vparand[i] = self.regul_limit
                    else:
                        vparand[i] = min(x, self.regul_limit)
                else:
                    vparand[i] = x

            if self.regul_smoothing_coef != 0.0:
                vparand = np.dot(vparand, self.regul_smoothing_matrix)

        for i in range(outputs):
            V[i, i] *= 1.0 + vparand[i]

        if self.regul_off_diag_correction:
            origoutputs = self.instrument_matrix.shape[1]
            for i in range(outputs):
                for j in range(outputs):
                    if i != j:
                        V[i, j] /= 1 + sqrt(vparand[i] * vparand[j]) / origoutputs

        Vcinv = scipy.linalg.inv(V)

        intermResult = np.dot(Vcinv, Wy)

        return intermResult, Vcinv, vparand

    def invert(
        self, in_sig: numpy.ndarray, in_var: numpy.ndarray, no_post=False
    ) -> (numpy.ndarray, numpy.ndarray):
        self.prepare()

        in_sig = np.array(in_sig, copy=True)
        in_var = np.array(in_var, copy=True)

        num_in = self.instrument_matrix.shape[0]
        num_out = self.instrument_matrix.shape[1]

        if (num_in != in_sig.size) or (num_in != in_var.size):
            raise InversionException(
                "Instrument matrix does not match signal vector size"
            )

        for i in range(num_in):
            v = in_sig[i]
            if isnan(v):
                in_sig[i] = 0.0
                in_var[i] = 10000000000.0

        # print "Initial:"
        values, variances, vparand = self.subinvert(
            self.instrument_matrix, in_sig, in_var
        )

        # concsum = init_values.sum()
        # totConcWeights = np.empty()

        indexmap = list(range(num_out))

        it_values = values
        it_variances = variances

        instrument_matrix = self.instrument_matrix

        for nriter in range(self.neg_remove_iterations):
            # print "Iteration: {}".format(nriter)
            minind = np.argmin(it_values)

            if it_values[minind] >= 0.0:
                break

            absminind = indexmap[minind]
            del indexmap[minind]

            it_instrument_matrix = instrument_matrix[:, indexmap]
            it_vparand = vparand[indexmap]

            it_values, it_variances, tmpvparand = self.subinvert(
                it_instrument_matrix, in_sig, in_var, it_vparand
            )

            values[absminind] = 0.0
            values[indexmap] = it_values

        # print "final values", fmt(values)

        if not (self.post_matrix is None or no_post):
            if self.post_matrix.shape[0] != len(values):
                raise InversionException(
                    "Post-transform matrix size does not match intermediate vector size"
                )

            result_values = np.dot(values, self.post_matrix)
            result_covariances = np.dot(
                self.post_matrix.T, np.dot(variances, self.post_matrix)
            )

        else:
            result_values = values
            result_covariances = variances

        if self.smoothing_coef > 0.0:
            result_values = np.dot(result_values, self.smoothing_matrix)
            result_covariances = np.dot(
                self.smoothing_matrix, np.dot(result_covariances, self.smoothing_matrix)
            )

        return result_values, result_covariances


def make_smoothing_matrix(size: int, coef: float):
    sm = np.zeros((size, size))

    coef += 1.0

    icoef = int(coef)

    v = np.empty((icoef + 1,))

    v[0] = 1.0
    vsum = 0.0

    for i in range(1, icoef + 1):
        x = cos(pi / coef * i) / 2.0 + 0.5
        v[i] = x
        vsum += x

    v /= (vsum * 2.0) + 1.0

    def vc(ind):
        ind = abs(ind)
        if ind > icoef:
            return 0.0
        else:
            return v[ind]

    for i in range(size):
        for j in range(size):
            sm[i, j] = vc(i - j) + vc(i + j + 1) + vc(2 * size - (i + j + 1))

    return sm


def save_inverter(
    outfilename, spectrum_name, model_parameters, inverter_options, model_result
):
    matrix_mult = model_parameters["matrix_mult"]

    if model_parameters["ion_mode"]:
        invtype = "mobility"
        invunit = "cm^2/s/V"
    else:
        invtype = "radius"
        invunit = "nm"

    instrument_matrix = model_result["instrument_matrix"]
    elsp_nd_matrix = model_result["elsp_nd_matrix"].T
    nd_sizes = model_result["nd_sizes"]

    model_parameters = copy.deepcopy(model_parameters)
    all_ndarray_tolist(model_parameters)

    doc = {
        "name": spectrum_name,
        "instrument": {
            "rows": instrument_matrix.shape[0],
            "cols": instrument_matrix.shape[1],
            "mult": matrix_mult,
            "data": (instrument_matrix * matrix_mult).tolist(),
        },
        "posttransform": {
            "rows": elsp_nd_matrix.shape[0],
            "cols": elsp_nd_matrix.shape[1],
            "data": elsp_nd_matrix.tolist(),
        },
        "xpoints": {
            "type": invtype,
            "unit": invunit,
            "points": (nd_sizes / model_parameters["scaling"]).tolist(),
        },
        "yunit": "1/cm^3",
        "model_parameters": model_parameters,
    }
    doc.update(inverter_options)

    out = open(outfilename, "w")
    yaml.safe_dump(doc, out, width=300, indent=4)

    return doc


def load_inverter(stream) -> Inverter:
    doc = yaml.safe_load(stream)
    inverter = Inverter()
    for key, fn in _INVERTER_ATTRIBUTES:
        if key in doc:
            setattr(inverter, key, fn(doc[key]))

    inverter.instrument_matrix = (
        np.array(doc["instrument_matrix"]["data"]) / doc["instrument_matrix"]["mult"]
    )
    inverter.post_matrix = np.array(doc["posttransform"]["data"])
    xp = doc["xpoints"]
    xtype = xp["type"]
    inverter.xpoints = np.array(xp["points"])
    inverter.xtype = xtype

    mil = millikan.Millikan()
    if xtype == "mobility":
        inverter.mobilities = inverter.xpoints
        inverter.diameters = mil.ktod(inverter.mobilities / 1e4) * 1e9
    elif xtype == "radius":
        inverter.diameters = inverter.xpoints * 2
        inverter.mobilities = mil.dtok(inverter.diameters / 1e9) * 1e4

    inverter.xunit = xp["unit"]
    inverter.yunit = doc["yunit"]
    return inverter


def all_ndarray_tolist(dic):
    for k, v in dic.iteritems():
        if isinstance(v, np.ndarray):
            dic[k] = v.tolist()
        elif isinstance(v, dict):
            all_ndarray_tolist(v)

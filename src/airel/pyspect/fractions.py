import math
from typing import Union

import numpy as np
import numpy.typing as npt


def nd_fraction_matrix(
    xpoints: npt.ArrayLike, limits: tuple[Union[None, float], Union[None, float]]
) -> npt.ArrayLike:
    log_xpoints = np.log10(xpoints)
    lowlim, uplim = limits
    log_limits = (
        None if lowlim is None else math.log10(lowlim),
        None if uplim is None else math.log10(uplim),
    )
    return nd_fraction_matrix_log(log_xpoints, log_limits)


def nd_fraction_matrix_log(
    log_xpoints: npt.ArrayLike,
    log_limits: tuple[Union[None, float], Union[None, float]],
) -> npt.ArrayLike:
    is_reversed = log_xpoints[0] > log_xpoints[-1]
    npoints = len(log_xpoints)

    matrix = np.zeros((npoints,))
    if is_reversed:
        uplim, lowlim = log_limits
        log_xpoints = log_xpoints[::-1]
    else:
        lowlim, uplim = log_limits

    if lowlim is None:
        lowlim = log_xpoints[0]

    if uplim is None:
        uplim = log_xpoints[-1]

    for i, (x0, x1) in enumerate(zip(log_xpoints[:-1], log_xpoints[1:])):
        if (lowlim > x1) or (uplim < x0):
            continue

        if lowlim < x0:
            w0 = 0.0
            rx0 = x0
        else:
            w0 = (lowlim - x0) / (x1 - x0)
            rx0 = x0 * (1 - w0) + x1 * w0

        if uplim > x1:
            w1 = 0.0
            rx1 = x1
        else:
            w1 = (x1 - uplim) / (x1 - x0)
            rx1 = x1 * (1 - w1) + x0 * w1

        matrix[i] += (rx1 - rx0) * (1 - w0 + w1) / 2
        matrix[i + 1] += (rx1 - rx0) * (1 - w1 + w0) / 2

    if is_reversed:
        return matrix[::-1]
    else:
        return matrix

# Util_emis3D.py
"""
Contains scaling and residual functions used by emis3D.

Note: All fitting is done centered around zero, dphi = phi - mu, where
mu is the injection location

Written by JLH Aug. 2025
"""

import numpy as np

from main.Util import convert_arrays_to_list
from scipy.special import i0
from lmfit import minimize


def _exp(dphi: np.ndarray, kappa: float = 0.0) -> np.ndarray:
    """Exponential function"""
    return np.exp(-1.0 * kappa * (dphi**2))


def scale_exp(A: float, B: float, dphi: np.ndarray) -> np.ndarray:
    raw = _exp(dphi, B)
    normalized = raw / _exp(np.zeros(1), B)
    return A * normalized


def scale_linear(A: float, B: float, dphi: np.ndarray) -> np.ndarray:
    return A * dphi + B


def scale_constant(A: float, dphi: np.ndarray) -> np.ndarray:
    return A * np.ones(dphi.shape[0])


def _von_mises(dphi: np.ndarray, kappa: float = 0.0) -> np.ndarray:
    """Von Mises distribution, normalized"""
    return np.exp(kappa * np.cos(dphi)) / (2.0 * np.pi * i0(kappa))


def von_mises_amplitude(A: float, B: float, dphi: np.ndarray) -> np.ndarray:
    """
    Apply scaling to Von Mises distribution, while ensuring that the
    endpoints are equal.

    left : theta-mu in (-pi, 0], counterClock
    right : theta-mu in (0, pi], clockwise
    """

    # --- Normalize VM so the value at mu = 1.0 (theta = mu -> dphi = 0)
    raw = _von_mises(dphi, B)
    normalized = raw / _von_mises(np.zeros(1), B)
    return A * normalized


def scale_wrapper(
    a: float,
    b: float,
    phi: np.ndarray,
    mu: float = 0.0,
    scale_def: str | None = None,
    emissionName: str | None = None,
    dphi: np.ndarray | None = None,
    numRevolutions: float = 1.0,
):
    """
    Wrapper for the scale function; returns a scaling factor array based on
    the chosen scale_def.

    Parameters
    ----------
    a             : Amplitude of the scaling function
    b             : Shape parameter used by most scaling functions
    phi           : Toroidal locations of the bolometers (radians)
    mu            : Injection location (radians)
    scale_def     : One of 'exponential', 'linear', 'constant', 'von_mises'
    emissionName  : Name of the emission distribution (e.g. 'clockwise')
    dphi          : Pre-computed dphi; calculated from phi/mu if None
    numRevolutions: Number of toroidal revolutions for helical distributions
    """

    # --- Find dphi
    if dphi is None:
        dphi = find_dphi(
            phi, mu, emissionName=str(emissionName), numRevolutions=numRevolutions
        )

    # --- Set small values of a to zero
    if a < 0.01:
        a = 0

    if scale_def == "exponential":
        return scale_exp(a, b, dphi)
    elif scale_def == "linear":
        return scale_linear(a, b, dphi)
    elif scale_def == "constant":
        return scale_constant(a, dphi)
    elif scale_def == "von_mises":
        return von_mises_amplitude(a, b, dphi)
    else:
        return np.ones(phi.shape[0])


def find_dphi(
    phi: np.ndarray,
    mu: float = 0.0,
    emissionName: str = "",
    scale: bool = True,
    numRevolutions: float = 1.0,
) -> np.ndarray:
    """
    Finds the change in toroidal angle between phi and mu.
    Example: phi = 220, mu = 100, dphi = 120 or 240 depending on the emssionName input.


    For helical distributions the result is also scaled by 0.5 (or
    1 / (2 * numRevolutions)) so that the range maps to [-pi, pi].

    Parameters
    ----------
    phi           : Toroidal locations of the bolometers (radians)
    mu            : Injection location (radians)
    emissionName  : 'clockwise', 'counterClock', or anything else for minimum
                    angular distance
    scale         : Whether to scale helical dphi into [-pi, pi]
    numRevolutions: Total revolutions of the helical distribution

    Returns
    -------
    dphi : np.ndarray
    """

    two_pi = 2 * np.pi

    # --- Find if we need to add 2 pi to the end result
    add_2pi = phi > two_pi

    phi = phi % two_pi
    mu = mu % two_pi

    cw = (mu - phi) % two_pi  # clockwise distance
    ccw = (phi - mu) % two_pi  # counter-clockwise distance

    # --- Correct for values that are greater than 2pi
    ccw[add_2pi] += two_pi
    cw[add_2pi] += two_pi

    # --- Helical distributions go a full revolution around the machine. Example:
    # counter-clock goes from the injection location back to the injection location
    # So we need to scale phi down to +/- pi for the fit, and scale it back up after
    # the fit
    #
    # We also need to account for the total number of revolutions the helical distribution
    # makes around the machine.
    if "clockwise" in emissionName:
        return -cw / (2.0 * numRevolutions) if scale else -cw
    elif "counterClock" in emissionName:
        return ccw / (2.0 * numRevolutions) if scale else ccw

    # --- For non-helical distributions, return the shorter angular distance
    return ccw if sum(ccw) < sum(cw) else -cw


def residual(
    pars,
    data_dict,
    synthetic_dict,
    scale_def: str | None = None,
    boloNames=None,
    residual: bool = True,
):
    """
    Computes the residual (or the scaled synthetic data) for the minimizer.

    When ``residual=True`` returns a flat list of (data - model) values for
    each bolometer channel (LMFIT squares these internally).
    When ``residual=False`` returns a dict of scaled synthetic arrays keyed
    by emissionName → bolometer-group index. This is used to make synthetic data
    in Emis3D._post_process_fit_arrangement()
    """

    a = 0.0
    b = 0.0
    params = pars.valuesdict()
    mu = float(synthetic_dict["injectionLocation_rad"])

    # --- Find the total synthetic emission
    temp_ = {}  # accumulated model signal per bolometer group
    data = {}  # per-emission scaled synthetic (returned when residual=False)

    for emissionName in synthetic_dict["emissionNames"]:
        # --- Find the number of revolutions the helical distribution makes,
        # it will return 0 for non-helical distributions
        if "clockwise" in emissionName or "counterClock" in emissionName:
            numRevolutions = len(synthetic_dict["emissionNames"]) / 2.0
        else:
            numRevolutions = 0.0

        data[emissionName] = {}
        # --- Get the new scale factor for the normal runs
        if boloNames is None:
            # --- Hard-coded parameter names... not ideal
            a = params[f"a_{synthetic_dict['injectionLocation']}"]
            if "clockwise" in emissionName:
                b = params[f"b_clockwise_{synthetic_dict['injectionLocation']}"]
            elif "counterClock" in emissionName:
                b = params[f"b_counterClock_{synthetic_dict['injectionLocation']}"]
            else:
                b = params[f"b_{emissionName}_{synthetic_dict['injectionLocation']}"]

        # --- Loop over each bolometer group
        for ii in range(len(synthetic_dict[emissionName]["data"])):

            if boloNames is not None:
                a = params[f"{boloNames[ii]}"]
                b = 0.0

            phi = np.array(synthetic_dict[emissionName]["scaleFactor"][ii])

            # --- Return the scale factor
            scale_ = scale_wrapper(
                a,
                b,
                phi=phi,
                mu=mu,
                scale_def=scale_def,
                emissionName=emissionName,
                numRevolutions=numRevolutions,
            )

            synth_ = np.array(synthetic_dict[emissionName]["data"][ii])
            # synth_error = np.array(synthetic_dict[emissionName]["data_error"][ii])

            if ii not in temp_:
                temp_[ii] = np.zeros(len(scale_))

            temp_[ii] += scale_ * synth_
            data[emissionName][ii] = scale_ * synth_

    if residual and data_dict is not None:
        res = []
        # --- Loop over each bolometer group
        for ii in temp_:
            data_ = np.array(data_dict["observed"][ii].copy())
            data_error = np.array(data_dict["observed_error"][ii].copy())

            # --- Make sure there are no zeros in the error
            data_error[data_error <= 1.0e-6] = 1.0e-6

            # --- Ignore channels with non-positive observed values
            bad_indices = np.where(data_ <= 0)[0]
            temp_[ii][bad_indices] = data_[bad_indices]

            # LMFIT minimises the sum of squares, so we return the raw residual
            numerator = data_ - temp_[ii]
            res.extend(convert_arrays_to_list(numerator))  # / data_error)

        return res
    else:
        return data


def runParallel(job):
    """Thin wrapper around residual minimisation for use with ProcessPoolExecutor."""
    boloNames = None
    res_ = True
    fit_index, pars, data_dict, synth_dict, scale_def = job

    fit = minimize(
        residual,
        pars,
        args=(
            data_dict,
            synth_dict,
            scale_def,
            boloNames,
            res_,
        ),
        method="leastsq",
    )

    return fit_index, fit


def error_exponential(
    signal: np.ndarray,
    max_signal: float,
    scale_factor: float = 1.0,
    decay: float = 4.0,
) -> np.ndarray:
    """
    Exponential decay error model.

    error ≈ scale_factor at signal = 0
    error ≈ 0            at signal = max_signal
    """
    signal = np.clip(signal, 0.0, None)
    return scale_factor * np.exp(-decay * signal / max_signal)


def error_inverse(
    signal: np.ndarray,
    max_signal: float,
    scale_factor: float = 1.0,
    _floor: float = 1.0e-12,
) -> np.ndarray:
    """
    Hyperbolic error model: error = scale_factor * (max_signal / signal).
    """
    signal = np.clip(signal, _floor, None)
    return scale_factor * (max_signal / signal)


def error_inv_sqrt(
    signal: np.ndarray,
    max_signal: float,
    scale_factor: float = 1.0,
    _floor: float = 1.0e-12,
) -> np.ndarray:
    """
    Inverse square-root (Poisson / shot-noise) error model.

    error = scale_factor * sqrt(max_signal / signal)
    error = 1.0 at peak signal.
    """
    signal = np.clip(signal, _floor, None)
    return scale_factor * np.sqrt(max_signal / signal)

# Util_emis3D.py
"""
Contains scaling and residual functions used by emis3D.

Note: All fitting is done centered around zero, dphi = phi - mu, where
mu is the injection location

Written by JLH Aug. 2025

TODO: Account for rev1, rev2, etc. for the helical distributions. The dphi
"""

import logging
import re
import numpy as np

from main.Util import convert_arrays_to_list
from scipy.special import i0
from lmfit import minimize

logger = logging.getLogger(__name__)


def _exp(dphi: np.ndarray, kappa: float = 0.0) -> np.ndarray:
    """Exponential function. Peak = 1.0 at dphi = 0."""
    return np.exp(-1.0 * kappa * (dphi**2))


def scale_exp(A: float, B: float, dphi: np.ndarray) -> np.ndarray:
    """Gaussian (exponential) scaling. Peak is A at dphi = 0"""
    return A * _exp(dphi, B)


def scale_linear(
    A: float,
    B: float,
    dphi: np.ndarray,
) -> np.ndarray:
    """
    Triangular profile centered at mu.

    Peak = B at dphi = 0.
    """
    return B - np.abs(A) * dphi


def scale_constant(A: float, dphi: np.ndarray) -> np.ndarray:
    return A * np.ones(dphi.shape[0])


def von_mises_amplitude(A: float, B: float, dphi: np.ndarray) -> np.ndarray:
    """
    Apply scaling to Von Mises distribution, while ensuring that the
    endpoints are equal.

    left : theta-mu in (-pi, 0], counterClock
    right : theta-mu in (0, pi], clockwise
    """

    return A * np.exp(B * (np.cos(dphi) - 1.0))


def scale_wrapper(
    a: float,
    b: float,
    phi: np.ndarray,
    mu: float = 0.0,
    scale_def: str | None = None,
    emissionName: str | None = None,
    dphi: np.ndarray | None = None,
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
            phi, mu, emissionName=str(emissionName), 
        )

    if scale_def == "exponential":
        return scale_exp(a, b, dphi)
    elif scale_def == "linear":
        return scale_linear(a, b, dphi)
    elif scale_def == "constant":
        return scale_constant(a, dphi)
    elif scale_def == "von_mises":
        return von_mises_amplitude(a, b, dphi)
    else:
        return np.ones(dphi.shape[0])


def find_dphi(
    phi: np.ndarray,
    mu: float = 0.0,
    emissionName: str = "",
) -> np.ndarray:
    """
    Finds the change in toroidal angle between phi and mu.

    Always returns the minimum distance for non-Helical distributions

    Examples
    --------
        phi = 220, mu = 100
        dphi = 120 for counterClock and non-Helical distributions
        dphi = 240 for clockwise Helical distributions

        phi = -370, mu = 30
        dphi = 40 + 360 = 400 for counterClock Helical and non-Helical distributions
        dphi = 320 + 360 = 680 for clockwise Helical distributions

        phi = 275, mu = 260
        dphi = 345 for counterClock Helical
        dphi = 15 for clockwise Helical and non-Helical distributions

        
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

    if emissionName is None:
        logger.warning(
            "Emission name in Util_emis3D.find_dphi is None."
        )
        return None

    phi = np.asarray(phi, dtype=float)

    two_pi = 2.0 * np.pi

    # Number of full revolutions represented by phi
    n_wraps = np.floor(np.abs(phi) / two_pi).astype(int)

    # exact multiples of 360 should not count the final wrap
    n_wraps = np.where(
        (np.abs(phi) > 0) & (np.mod(np.abs(phi), two_pi) == 0),
        n_wraps - 1,
        n_wraps,
    )

    wraps = n_wraps * two_pi

    phi_mod = phi % two_pi
    mu_mod = mu % two_pi

    # Positive directional distances
    ccw = (phi_mod - mu_mod) % two_pi
    cw = (mu_mod - phi_mod) % two_pi

    # Add back complete revolutions
    ccw += wraps
    cw += wraps

    # --- Account for 2nd or more revolutions around the tokamak
    rev_match = re.search(r"_rev(\d+)", emissionName)
    n_rev = int(rev_match.group(1)) if rev_match else 0
    rev_offset = n_rev * two_pi
        
    if "clockwise" in emissionName:
        return cw + rev_offset

    if "counterClock" in emissionName:
        return ccw + rev_offset

    return np.minimum(ccw, cw)


def loc_tag(injectionLocation) -> str:
    """Canonical tag for LMFIT parameter names ('a_240', 'b_ ... _ 240', etc.)
    Float injection locations must be int to agree with the creation"""
    return str(int(round(float(injectionLocation))))

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

    # --- Check to see if the peak radiation location can be varied
    mu = float(synthetic_dict["injectionLocation_rad"])
    if "peak_rad_loc" in params:
        mu = float(params["peak_rad_loc"])

    # --- Find the total synthetic emission
    temp_ = {}  # accumulated model signal per bolometer group
    data = {}  # per-emission scaled synthetic (returned when residual=False)

    for emissionName in synthetic_dict["emissionNames"]:

        # --- Find the number of revolutions the helical distribution makes
        numRevolutions = 1.0

        if "clockwise" in emissionName or "counterClock" in emissionName:
            if "info" in synthetic_dict:
                if "numTransists" in synthetic_dict["info"]:
                    numRevolutions = synthetic_dict["info"]["numTransists"]

        data[emissionName] = {}
        tag = loc_tag(synthetic_dict['injectionLocation'])
        # --- Get the new scale factor for the normal runs
        if boloNames is None:
            a = params[f"a_{tag}"]
            b = params[f"b_{emissionName}_{tag}"]

        # --- Loop over each bolometer group
        for ii in range(len(synthetic_dict[emissionName]["data"])):

            # --- Different parameter names when doing the cross-calibration
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
            res.extend(convert_arrays_to_list(numerator / data_error))

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


def signal_error(
    type: str,
    signal: np.ndarray,
    max_signal: float,
    scale_factor: float = 1.0,
):
    """Wrapper for the error function"""

    type_ = type.upper()

    if type_ == "EXPONENTIAL":
        return error_exponential(signal, max_signal, scale_factor)
    elif type_ == "INVERSE":
        return error_inverse(signal, max_signal, scale_factor)
    elif type_ == "INVERSE SQUARE":
        return error_inv_sqrt(signal, max_signal, scale_factor)
    elif type_ == 'CONSTANT':
        return error_constant(signal, max_signal)


def error_constant(
        signal: np.ndarray,
        max_signal: float,
        _floor: float = 1.0e-12,
        fraction: float =  0.1 # 10%
) -> np.ndarray:
    """Returns fraction * max_signal for each signal"""
    signal = np.asarray(signal)
    signal = np.clip(signal, _floor, None)
    
    return fraction * max_signal / signal

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


def print_intro() -> None:

    print("""

    ███████╗███╗   ███╗██╗███████╗██████╗ ██████╗ 
    ██╔════╝████╗ ████║██║██╔════╝╚════██╗██╔══██╗
    █████╗  ██╔████╔██║██║███████╗ █████╔╝██║  ██║
    ██╔══╝  ██║╚██╔╝██║██║╚════██║ ╚═══██╗██║  ██║
    ███████╗██║ ╚═╝ ██║██║███████║██████╔╝██████╔╝
    ╚══════╝╚═╝     ╚═╝╚═╝╚══════╝╚═════╝ ╚═════╝
    """)

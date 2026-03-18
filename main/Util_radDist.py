# Util_RadDist.py
"""
File contains defintions used while creating radDists

Written by JLH Aug. 2025

TODO:
1. Should we normalize the bivariate distributions to 2 pi? Since we integrate
over the whole tokamak while calculating the radiated power?
"""


import random

import numpy as np

import main.radDist as radDist
import main.Util as Util
from typing import Optional


def radDist_ElongatedRing_parallel(input) -> None:
    """
    Computes Elongated Ring radDists based on input rzArray
    """
    rzArray = input[0]
    config = input[1]
    elongatedRing = radDist.ElongatedRing(
        startR=rzArray[0], startZ=rzArray[1], config=config
    )
    elongatedRing.build()

    print(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}m, z = {rzArray[1]:.2f}m"
    )


def radDist_ElongatedRing_parallel_return_radDist(input):
    """
    Computes Elongated Ring radDists based on input rzArray
    """
    rzArray = input[0]
    config = input[1]
    elongatedRing = radDist.ElongatedRing(
        startR=rzArray[0], startZ=rzArray[1], config=config
    )
    elongatedRing.build()

    print(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}m, z = {rzArray[1]:.2f}m"
    )
    return elongatedRing


def radDist_Helical_parallel(input) -> None:
    """
    Computes helical radDists based on input rzArray
    """

    rzArray = input[0]
    config = input[1]
    helical = radDist.Helical(startR=rzArray[0], startZ=rzArray[1], config=config)
    helical.build()

    print(f"DONE with helical radDist, R = {rzArray[0]:.2f}m, z = {rzArray[1]:.2f}m")


def radDist_Helical_parallel_return_radDist(input):
    """
    Computes helical radDists based on input rzArray
    """

    rzArray = input[0]
    config = input[1]
    helical = radDist.Helical(startR=rzArray[0], startZ=rzArray[1], config=config)
    helical.build()

    print(f"DONE with helical radDist, R = {rzArray[0]:.2f}m, z = {rzArray[1]:.2f}m")
    return helical


def radDist_SquareTube_parallel(input) -> None:
    """
    Computes Square Tube radDists based on input rzArray
    """
    rzArray = input[0]
    config = input[1]
    squareTube = radDist.SquareTube(startR=rzArray[0], startZ=rzArray[1], config=config)
    squareTube.build()

    print(f"DONE with squareTube radDist, R = {rzArray[0]:.2f}m, z = {rzArray[1]:.2f}m")


def radDist_SquareTube_parallel_return_radDist(input):
    """
    Computes Square Tube radDists based on input rzArray
    """
    rzArray = input[0]
    config = input[1]
    squareTube = radDist.SquareTube(startR=rzArray[0], startZ=rzArray[1], config=config)
    squareTube.build()

    print(f"DONE with squareTube radDist, R = {rzArray[0]:.2f}m, z = {rzArray[1]:.2f}m")
    return squareTube


def callRZGridTokamak(tokamak, numRgrid=30, numZgrid=15) -> np.ndarray:
    """
    Calls createRZgrid using the tokamak class as an input
    """
    rLimits = [tokamak.wall["minr"], tokamak.wall["maxr"]]
    zLimits = [tokamak.wall["minz"], tokamak.wall["maxz"]]

    rzarray = createRZGrid(
        rLimits=rLimits,
        zLimits=zLimits,
        numRgrid=numRgrid,
        numZgrid=numZgrid,
        wallcurve=tokamak.wall["wallcurve"],
    )
    return rzarray


def createRZGrid(
    rLimits=[0, 1], zLimits=[-1, 1], numRgrid=30, numZgrid=15, wallcurve=None
) -> np.ndarray:
    """
    Creates a equally spaced R, z grid within the given limits.
    The program will eliminate points outside of the wall, if
    the wallcurve is specified (from the Tokamak class)

    INPUTS

    rLimits :: List of R min and R max points
    zLimits :: List of z min and z max points
    wallcurve :: Created within the tokamak class when _load_first_wall() is called.
                 Type: path.Path(rzarray)
    numRgrid :: The number of equally spaced points within the R grid
    numZgrid :: The number of equally spaced points within the z grid
    """

    rarray = np.linspace(rLimits[0], rLimits[1], num=numRgrid)
    zarray = np.linspace(zLimits[0], zLimits[1], num=numZgrid)

    # we now have our RZ grid specified. We will now reduce the size by
    # omitting points outside the first wall
    rzarray = []
    for j in range(numZgrid):
        for i in range(numRgrid):
            if wallcurve is not None:
                # --- Only save the point if it is inside of the wall
                if wallcurve.contains_points([(rarray[i], zarray[j])]):
                    rzarray.append([rarray[i], zarray[j]])
            else:
                rzarray.append([rarray[i], zarray[j]])

    return np.array(rzarray)


def random_uniform_point_noVolume(Wallcurve, Minr, Maxr, Minz, Maxz):

    x = y = z = r = phi = None  # Initialize variables to ensure they are always defined
    success = 0
    while success == 0:
        x = random.uniform(-Maxr, Maxr)
        y = random.uniform(-Maxr, Maxr)
        z = random.uniform(Minz, Maxz)
        r, phi = Util.XY_To_RPhi(x, y)
        r = np.sqrt((x**2) + (y**2))

        if r < Minr or r > Maxr:
            pass
        elif Wallcurve.contains_points([(r, z)]):
            success = 1

    return x, y, z, r, phi


def bivariate_normal_elongated(
    R: np.ndarray,
    z: np.ndarray,
    R0: float = 0.0,
    z0: float = 0.0,
    elongation: float = 1.0,
    pol_sigma: float = 1.0,
    theta: float = 0.0,
):
    """
    Rotatable 2-D Gaussian distribution in the poloidal plane.

    Generalises bivariate_normal by allowing an elliptical (elongated)
    and rotated beam profile. Normalised such that ∫∫ f dR dz = 1.

    See: https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

    Parameters
    ----------
    R, z        : array-like — evaluation coordinates.
    R0, z0      : float      — centre of the distribution.
    elongation  : float      — standard deviation along the R axis. Must be > 0.
    pol_sigma    : float      — standard deviation along the z axis. Must be > 0.
    theta       : float      — rotation angle in degrees (counter-clockwise from R-axis).

    Returns
    -------
    np.ndarray — density at each (R, z) point.
    """

    # --- Make sure R and z are numpy arrays
    R = np.asarray(R)
    z = np.asarray(z)

    theta_rad = np.deg2rad(theta)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

    a = cos_t**2 / (2.0 * elongation**2) + sin_t**2 / (2.0 * pol_sigma**2)

    b = np.sin(2.0 * theta_rad) / (4.0 * elongation**2) - np.sin(2.0 * theta_rad) / (
        4.0 * pol_sigma**2
    )
    c = sin_t**2 / (2.0 * elongation**2) + cos_t**2 / (2.0 * pol_sigma**2)

    dR = R - R0
    dz = z - z0

    norm = 1.0 / (2.0 * np.pi * elongation * pol_sigma)
    exponent = -(a * dR**2 + 2.0 * b * dR * dz + c * dz**2)

    return norm * np.exp(exponent)


def bivariate_normal(
    R: np.ndarray,
    z: np.ndarray,
    R0: float = 0.0,
    z0: float = 0.0,
    pol_sigma: float = 1.0,
) -> np.ndarray:
    """
    Bivariate normal distribution in the poloidal plane.

    Normalised such that ∫∫ f dR dz = 1.

    Parameters
    ----------
    R, z      : array-like — evaluation coordinates.
    R0, z0    : float      — centre of the distribution.
    pol_sigma : float      — standard deviation (same in R and z). Must be > 0.

    Returns
    -------
    emis : np.ndarray — density at each (R, z) point.
    """

    # --- Ensure that R and z are numpy arrays
    R = np.asarray(R)
    z = np.asarray(z)

    norm = 1 / (2 * np.pi * pol_sigma**2)
    exponent = -0.5 * ((R - R0) ** 2 + (z - z0) ** 2) / pol_sigma**2

    return norm * np.exp(exponent)


def bivariate_normal_isodensity_points(
    R0: float,
    z0: float,
    sigma_target: float,
    sigma_kernel: float,
    n: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n source points and weights such that:

        sum_i = w_i * bivariate_normal(R, z, R0=R0_i, z0=z0_i, pol_sigma=sigma_kernel)
        ≈ bivariate_normal(R, z, R0=R0, z0=z0, pol_sigma=sigma_target)


    Parameters
    ----------
    R0, z0        : float — centre of the target distribution.
    sigma_target  : float — std dev of the target Gaussian (σ_T). Must be > sigma_kernel.
    sigma_kernel  : float — std dev used in each bivariate_normal call (σ_k).
    n             : int   — number of source points.
    seed          : int, optional — RNG seed for reproducibility.

    Returns
    -------
    points  : np.ndarray, shape (n, 2) — source (R, z) coordinates.
    weights : np.ndarray, shape (n,)   — equal weights summing to 1.

    Raises
    ------
    ValueError if sigma_kernel >= sigma_target (convolution can't shrink a Gaussian).
    """
    if sigma_kernel >= sigma_target:
        raise ValueError(
            f"sigma_kernel ({sigma_kernel}) must be strictly less than "
            f"sigma_target ({sigma_target})."
        )

    sigma_source = np.sqrt(sigma_target**2 - sigma_kernel**2)

    rng = np.random.default_rng(seed)
    points = rng.normal(loc=[R0, z0], scale=sigma_source, size=(n, 2))
    weights = np.ones(n) / n

    return points, weights

# newHelical_radDist_tester.py
"""
Testing out a new newHelical radDistok. This one should expand as it moves away from
the creation point (aka injection location)

Perhaps start out with many field lines, with each of those having a Gaussian distribution?
Then we add them together to get the radDist
"""


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np

import main.Util_radDist as Util_radDist
from main.Globals import *
from main.Tokamak import Tokamak
from main.Util import config_loader


tokamakName = "DIII-D"
startPhi = 225
R0 = 2.029
z0 = 0.409


colors = ["purple", "red", "blue", "green", "purple", "orange"]


def circle_points(R, z, dr, n_points):
    """
    Generate points around (R, z) in a circle of radius dr.

    Parameters:
        R (float): center R coordinate
        z (float): center z coordinate
        dr (float): radius of the circle
        n_points (int): number of points to generate

    Returns:
        list of tuples: [(R1, z1), (R2, z2), ...]
    """
    R_vals = []
    z_vals = []
    for i in range(n_points):
        theta = 2 * np.pi * i / n_points
        R_vals.append(R + dr * np.cos(theta))
        z_vals.append(z + dr * np.sin(theta))

    return R_vals, z_vals


def pellet_initial_parameters(csp=False):
    """
    Gives the basic SPI trajector parameters
    """
    data = {}
    if csp:
        # --- CSP trajectory parameters
        data["R_OUT"] = 2.35  # 2.249
        data["Z_OUT"] = 0.0

        # SPI breaker tube angle down (degrees)
        data["THETA"] = 0
        data["DISP"] = 2

        # SPI length (arbitrary, can increase or decrease)
        data["LENGTH"] = 0.8

    else:
        # --- SPI trajectory parameters
        data["R_OUT"] = 2.284
        # SPI breaker tube tip Z (outer wall)
        data["Z_OUT"] = 0.6845

        # SPI breaker tube angle down (degrees)
        data["THETA"] = 47.3
        data["DISP"] = 15.0

        # SPI length (arbitrary, can increase or decrease)
        data["LENGTH"] = 0.8

    data["Z_IN"] = data["Z_OUT"] - data["LENGTH"] * np.sin(np.deg2rad(data["THETA"]))
    data["R_IN"] = data["R_OUT"] - data["LENGTH"] * np.cos(np.deg2rad(data["THETA"]))

    # --- Find the upper and lower scatter points
    dz = data["LENGTH"] * np.sin((np.deg2rad(data["THETA"] - data["DISP"])))
    dr = data["LENGTH"] * np.cos((np.deg2rad(data["THETA"] - data["DISP"])))
    data["Z_IN_UPPER"] = data["Z_OUT"] - dz
    data["R_IN_UPPER"] = data["R_OUT"] - dr

    dz = data["LENGTH"] * np.sin((np.deg2rad(data["THETA"] + data["DISP"])))
    dr = data["LENGTH"] * np.cos((np.deg2rad(data["THETA"] + data["DISP"])))
    data["Z_IN_LOWER"] = data["Z_OUT"] - dz
    data["R_IN_LOWER"] = data["R_OUT"] - dr

    return data


tok = Tokamak(
    tokamakName=tokamakName,
    mode="Build",
    reflections=False,
    eqFileName="g184407.02100",
    loadBolometers=True,
    verbose=False,
)


sigma_target = 0.06
sigma_kernel = 0.05  # pol_sigma
n = 2_000
points, weights = Util_radDist.bivariate_normal_isodensity_points(
    R0, z0, sigma_target, sigma_kernel, n, seed=42
)
Rvalues = points[:, 0]  # + R0
zValues = points[:, 1]  # + z0


"""
Rvalues = points[:, 0]
zValues = points[:, 1]
plt.ion()
plt.figure()
plt.scatter(Rvalues, zValues)
"""


tok.set_fieldlines(
    startR=Rvalues,
    startZ=zValues,
    startPhi=np.deg2rad(startPhi),
    numTransists=1.0,
)

# --- Initial test at the injection location
if False:
    # --- Each field line will be a bivariate normal, let's caclculate the
    # scale factor for points that are not in the center
    emissionName = "counterClock_rev0"
    R, z = tok.find_RZ_Fline(
        str(startPhi),
        emissionName=emissionName,
        inputPhis=[np.deg2rad(startPhi)],
    )

    # ans = Util_radDist.bivariate_normal(R=R, R0=R[0], z=z, z0=z[0], pol_sigma=0.05)
    # scale = ans / ans[0]

    # R = [R[0], np.array(1.5)]
    # z = [z[0], np.array(0.0)]
    # --- Create a R, z grid. Find the total simulated radiation at that toroidal location
    RZarray = Util_radDist.callRZGridTokamak(tok, numRgrid=500, numZgrid=500)
    tot = np.zeros(len(RZarray))

    # --- This function will be replaced by the _evalulate call, using x, y, and z cherab inputs
    # instead of a pre-defined grid
    for (Ri, zi), w, R0, z0 in zip(points, weights, R, z):
        tot += w * Util_radDist.bivariate_normal(
            R=RZarray[:, 0], z=RZarray[:, 1], R0=R0, z0=z0, pol_sigma=sigma_kernel
        )

    f = plt.figure()
    ax = f.add_subplot(111)
    tok._plot_first_wall(ax)
    R_unique = RZarray[:, 0]
    z_unique = RZarray[:, 1]
    n_levels = 50

    cf = ax.tricontourf(R_unique, z_unique, tot, levels=n_levels, cmap="plasma")
    ax.tricontour(
        R_unique,
        z_unique,
        tot,
        levels=n_levels,
        colors="white",
        linewidths=0.4,
        alpha=0.4,
    )
    plt.colorbar(cf, ax=ax, label="Density")
    plt.show()


# --- Plot the field lines at the relevant locations
phis = [startPhi, 270, 270, 135]
groupNames = ["", "SX90PF", "SX90MF", "DISRADU"]
plt.ion()
f, ax = plt.subplots(figsize=(18, 8), ncols=4, nrows=1)

emissionNames = tok.fieldLines[str(startPhi)]["directionNames"]
# emissionNames = ["counterClock_rev0"]
for ii, ax_ in enumerate(ax.flat):

    # --- Find the simulated radiation at that location
    RZarray = Util_radDist.callRZGridTokamak(tok, numRgrid=500, numZgrid=500)
    tot = np.zeros(len(RZarray))

    for qq, emissionName in enumerate(emissionNames):
        R, z = tok.find_RZ_Fline(
            str(startPhi),
            emissionName=emissionName,
            inputPhis=[np.deg2rad(phis[ii])],
        )

        for (Ri, zi), w, R0, z0 in zip(points, weights, R, z):
            tot += w * Util_radDist.bivariate_normal(
                R=RZarray[:, 0], z=RZarray[:, 1], R0=R0, z0=z0, pol_sigma=sigma_kernel
            )

        R_unique = RZarray[:, 0]
        z_unique = RZarray[:, 1]
        n_levels = 50

        cf = ax_.tricontourf(R_unique, z_unique, tot, levels=n_levels, cmap="CMRmap_r")
        ax_.tricontour(
            R_unique,
            z_unique,
            tot,
            levels=n_levels,
            colors="white",
            linewidths=0.4,
            alpha=0.4,
        )

        tok._plot_first_wall(ax_)

        # --- Injection location
        if ii == 0:
            spi_path = pellet_initial_parameters(csp=False)
            ax_.plot(
                [spi_path["R_IN"], spi_path["R_OUT"]],
                [spi_path["Z_IN"], spi_path["Z_OUT"]],
                "-r",
            )
            ax_.plot(
                [spi_path["R_IN_UPPER"], spi_path["R_OUT"]],
                [spi_path["Z_IN_UPPER"], spi_path["Z_OUT"]],
                "-r",
            )
            ax_.plot(
                [spi_path["R_IN_LOWER"], spi_path["R_OUT"]],
                [spi_path["Z_IN_LOWER"], spi_path["Z_OUT"]],
                "-r",
            )

            ax_.plot(tok.gfile.rbbbs, tok.gfile.zbbbs, color="teal")
            ax_.set_title("Injection location")

        # --- Plot the initial points
        if False:
            for jj in range(0, len(R.flatten())):
                label = "__no_legend__"
                if jj == 0:
                    label = emissionName

                ax_.scatter(
                    R[jj], z[jj], color=colors[qq], marker="s", label=label, zorder=10
                )

        # --- Plot the bolometers
        if ii > 0:
            tok._plot_bolometers(ax_, boloGroupName=groupNames[ii])

ax[1].legend()
plt.tight_layout()
plt.show()

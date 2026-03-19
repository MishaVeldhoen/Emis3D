# radDistTester_DIII-D.py
"""
This program will group similar SXR arrays, then plot out
the chords, radDist contour plot, and the observed radiation
below.

It is currently specific to DIII-D
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np

import main.Util_radDist as Util_radDist
from main.Globals import *
import time
from main.Util import config_loader
import main.radDist as radDist


tokamakName = "DIII-D"
configFileName = "helical_config.yaml"  # "sqaureTube_config.yaml"  # "elongatedRing_config.yaml"  # "helical_config.yaml"
rzvalues = [2.0, 0.56]


# --- Create the radDist using only one point, we don't need to loop over everything
pathFileName = os.path.join(
    EMIS3D_INPUTS_DIRECTORY, tokamakName, "radDists", configFileName
)
config = config_loader(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")


# --- Update the configuration file
rzArray = np.array([rzvalues[0], rzvalues[1]])

config["polSigma"] = config["polSigmas"][0]
if "elongations" in config:
    config["elongation"] = config["elongations"][0]
if "rotationAngles" in config:
    config["rotationAngle"] = config["rotationAngles"][0]
arg_list = (rzArray, config)

config["setFieldLine"] = False
# --- Try to identify bottlenecks
t0 = time.time()
rD = radDist.Helical(startR=rzArray[0], startZ=rzArray[1], config=config)
print(f"Initilization: {time.time() - t0:.1f} s")

# --- Building the tokamak
t0 = time.time()
rD._build_tokamak(
    tokamakName=rD.info["tokamakName"],
    mode="Build",
    reflections=False,
    eqFileName=rD.info["eqFileName"],
)
print(f"Built tokamak: {time.time() - t0:.1f} s")

# --- Setting the field lines
t0 = time.time()
rD.setFieldLine()
print(f"Field lines created: {time.time() - t0:.1f} s")


# --- Going through rD.build()
t0 = time.time()
# rD.power_per_bin_calc()
print(f"Power per bin caclulated: {time.time() - t0:.1f} s")

t0 = time.time()
rD.data = {}
rD.bolos_observe()

print(f"Bolos observed in: {time.time() - t0:.1f} s")

"""
# helical.build()
phi = 225.0
RZarray = Util_radDist.callRZGridTokamak(rD.tokamak, numRgrid=500, numZgrid=500)
emiss = np.zeros(len(RZarray))

for emissionName in rD.info["emissionNames"]:
    temp = rD.calc_emissivity(
        RZarray[:, 0],
        RZarray[:, 1],
        phi=np.array([np.deg2rad(phi)]),
        emissionName=emissionName,
    )
    emiss += temp[emissionName]


f = plt.figure()
ax_ = f.add_subplot(111)
R_unique = RZarray[:, 0]
z_unique = RZarray[:, 1]
n_levels = 50

rD.tokamak._plot_first_wall(ax_)
cf = ax_.tricontourf(
    R_unique, z_unique, emiss, levels=n_levels, cmap="CMRmap_r"
)
ax_.tricontour(
    R_unique,
    z_unique,
    emiss,
    levels=n_levels,
    colors="white",
    linewidths=0.4,
    alpha=0.4,
)
plt.show()


# --- Create the radDist
if config["distType"] == "Helical":
    rD = Util_radDist.radDist_Helical_parallel(arg_list, return_result=True)
if config["distType"] == "HelicalRing":
    rD = Util_radDist.radDist_HelicalRing_parallel(arg_list, return_result=True)
elif config["distType"] == "ElongatedRing":
    rD = Util_radDist.radDist_ElongatedRing_parallel(arg_list, return_result=True)
elif config["distType"] == "SquareTube":
    rD = Util_radDist.radDist_SquareTube_parallel(arg_list, return_result=True)
else:
    raise RuntimeError(
        "Please have 'elongatedRing', 'helical', or 'SqureTube' in the configFileName"
    )

"""

# --- Plot everything ---
if rD is not None:
    rD.plotOverview(plot_etendue=["SX45F07"])

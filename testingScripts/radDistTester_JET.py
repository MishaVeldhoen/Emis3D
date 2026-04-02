# radDistTester_JET.py
"""
This program will group similar SXR arrays, then plot out
the chords, radDist contour plot, and the observed radiation
below.

It is currently specific to JET
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np

import main.Util_radDist as Util_radDist
from main.Globals import *
from main.Util import config_loader
from main.Emis3D import Emis3D

tokamakName = "JET"
configFileName = (
    "helical_config.yaml"  # helical_config.yaml, or elongatedRing_config.yaml
)
rzvalues = [2.897, 1.39]


plt.ion()

# --- Create the radDist using only one point, we don't need to loop over everything
pathFileName = os.path.join(
    EMIS3D_INPUTS_DIRECTORY, tokamakName, "radDists", configFileName
)
config = config_loader(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")


# --- Update the configuration file
rzArray = np.array([rzvalues[0], rzvalues[1]])

if "polSigmas" in config:
    config["polSigma"] = config["polSigmas"][0]
if "elongations" in config:
    config["elongation"] = config["elongations"][0]
if "rotationAngles" in config:
    config["rotationAngle"] = config["rotationAngles"][0]
arg_list = (rzArray, config)

# --- Create the radDist
if config["distType"] == "Helical":
    rD = Util_radDist.radDist_Helical_parallel(arg_list, return_result=True)
elif config["distType"] == "HelicalRing":
    rD = Util_radDist.radDist_HelicalRing_parallel(arg_list, return_result=True)
elif config["distType"] == "ElongatedRing":
    rD = Util_radDist.radDist_ElongatedRing_parallel(arg_list, return_result=True)
elif config["distType"] == "SquareTube":
    rD = Util_radDist.radDist_SquareTube_parallel(arg_list, return_result=True)
else:
    raise RuntimeError(
        "Please have 'elongatedRing', 'helical', 'HelicalRing', or 'SqureTube' in the configFileName"
    )


# --- Plot everything ---
if rD is not None:
    rD.plotOverview(plot_etendue=[""])

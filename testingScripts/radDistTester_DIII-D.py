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
from main.Tokamak import Tokamak
from main.Util import config_loader

tokamakName = "DIII-D"
configFileName = "184407_injectionLocation_225/helical_config.yaml" # "sqaureTube_config.yaml"  # "elongatedRing_config.yaml"  # "helical_config.yaml"  #
elongation = 0.2 # 2.0
polSigma = 0.3 # 0.05
rotationAngle = 0.0
rzvalues = [2.0, 0.56]


# --- Create the radDist using only one point, we don't need to loop over everything
pathFileName = os.path.join(
    EMIS3D_INPUTS_DIRECTORY, tokamakName, "radDists", configFileName
)
config = config_loader(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")

tok = Tokamak(
    tokamakName=tokamakName,
    mode="Build",
    reflections=False,
    loadBolometers=False,
    verbose=True,
)

rzArray = Util_radDist.callRZGridTokamak(
    tok,
    numRgrid=config["GRID"]["NumRStartGrid"],
    numZgrid=config["GRID"]["NumZStartGrid"],
)

# --- Old tokamak is not needed anymore
del tok

# --- Update the configuration file
rzArray[0] = [rzvalues[0], rzvalues[1]]
config["polSigma"] = polSigma
config["elongation"] = elongation
config["rotationAngle"] = rotationAngle
arg_list = [(val, config) for val in rzArray]
arg_list = arg_list[0]


# --- Decrease the number of sampling points used, to speed up the process
arg_list[1]["BOLOMETER_PROPS"] = {"pixelSamples": 100, "numProcessors": 1}


# --- Create the radDist
if config["distType"] == "Helical":
    rD = Util_radDist.radDist_Helical_parallel(arg_list, return_result=True)
elif config["distType"] == "ElongatedRing":
    rD = Util_radDist.radDist_ElongatedRing_parallel(arg_list, return_result=True)
elif config["distType"] == "SquareTube":
    rD = Util_radDist.radDist_SquareTube_parallel(arg_list, return_result=True)
else:
    raise RuntimeError(
        "Please have 'elongatedRing', 'helical', or 'SqureTube' in the configFileName"
    )

# --- Plot everything ---
if rD is not None:
    rD.plotOverview(plot_etendue=["SX45F07"])

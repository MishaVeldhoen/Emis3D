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
configFileName = "elongatedRing_config.yaml"  # "helical_config.yaml"  #
elongation = 0.1
polSigma = 0.1
rotationAngle = 0.0
rzvalues = [1.65, 0.0]


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
arg_list[1]["BOLOMETER_PROPS"] = {"pixelSamples": 500, "numProcessors": 1}


# --- Create the radDist
if config["distType"] == "Helical":
    rD = Util_radDist.radDist_Helical_parallel_return_radDist(arg_list)
elif config["distType"] == "ElongatedRing":
    rD = Util_radDist.radDist_ElongatedRing_parallel_return_radDist(arg_list)
else:
    raise RuntimeError("Please have 'elongatedRing' or 'helical' in the configFileName")

# --- Plot everything ---
rD.plotOverview()

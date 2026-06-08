# radDistTester_JET.py
"""
This program will group similar SXR arrays, then plot out
the chords, radDist contour plot, and the observed radiation
below.

It is currently specific to JET
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))

import numpy as np
import main.Util_radDist as Util_radDist
from main.Globals import EMIS3D_INPUTS_DIRECTORY
from main.Util import config_loader

tokamakName = "JET"
configFileName = (
    "helical_config.yaml"  # helical_config.yaml, or elongatedRing_config.yaml
)
rzvalues = [2.897, 1.39]


# --- Create the radDist using only one point, we don't need to loop over everything
pathFileName = EMIS3D_INPUTS_DIRECTORY / tokamakName / "radDists" / configFileName
config = config_loader(pathFileName)
print(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")


# --- Update the configuration file
rzArray = np.array([rzvalues[0], rzvalues[1]])

if "sigma_R_vals" in config:
    config["sigma_R"] = config["sigma_R_vals"][0]
if "sigma_z_vals" in config:
    config["sigma_z"] = config["sigma_z_vals"][0]
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

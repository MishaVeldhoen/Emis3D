# postProcessing_testing.py
"""
Tests the post-processing TPF calculation

"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from main.Emis3D import Emis3D
import main.Util_emis3D as Util_emis3D
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# NOTE: Program assumes you already have a fit done, and loads it

evalTime = 50.9556
tokamakName = "JET"
runConfigName = "95709/95709_runConfig.yaml"
verbose = True

t = Emis3D(
    tokamakName=tokamakName,
    runConfigName=runConfigName,
    verbose=verbose,
    initialize=True,
)

t._load_bestFits(path="/Users/plh/Documents/git/Emis3D/inputs/JET/runs/95709/95709_bestFits_50.9556.dill")

'''
# --- Calculates the radiation amplitude distribution from the best fit

Main goals after a run:
1. Calculate overall radiation at each toroidal angle
2. Determine TPF

To do this:
1. Re-build "scaling function" to multiply the radDist power per bin for each emission
2. Add each emission together


'''

def _bestFit_radDist_parameters(emissionName) -> tuple[np.ndarray, np.ndarray]:
    """Finds the starting and ending phi for a given radDist"""
    phi = t.bestFits[evalTime]['radDist'].data['toroidalRadiatedPower'][emissionName]['phi_array']
    P_pol = t.bestFits[evalTime]['radDist'].data['toroidalRadiatedPower'][emissionName]['P_pol']
    return np.array(phi), np.array(P_pol)

def _bestFit_radDist_scaling(emissionName) -> float:
    """Returns the pre-fit synthetic / data normalization factor"""
    norm_factor= t.bestFits[evalTime]['synthetic_dict'][emissionName]['scaleSynth']
    return norm_factor

if t.info is not None:

    # --- First definition:
    # 1. Calculate the toroidal scaling factor for each emissionName based on the best fit parameters
    """Calculates the radiation amplitude distribution from the best fit"""

    # --- Create an empty dictionary
    t.bestFits[evalTime]["radiation_distribution"] = {}
    rad_distribution = t.bestFits[evalTime]["radiation_distribution"]
    emissionNames = t.bestFits[evalTime]["synthetic_dict"]["emissionNames"]

    scale_def = t.info["scale_def"]
    params = t.bestFits[evalTime]["fit"].params.valuesdict()

    # --- Check to see if the peak radiation location can be varied
    mu = t.bestFits[evalTime]["synthetic_dict"]["injectionLocation_rad"]
    if "peak_rad_loc" in params:
        mu = float(params["peak_rad_loc"])

    # Value at the end of each tag in params
    inj_loc_tag = Util_emis3D.loc_tag(t.bestFits[evalTime]["synthetic_dict"]["injectionLocation"])


    for emissionName in emissionNames:
        rad_distribution[emissionName] = {}

        # Find the phi limits that the radDist was created with
        rD_phi, rD_P_pol = _bestFit_radDist_parameters(emissionName)
        rD_scale = _bestFit_radDist_scaling(emissionName)

        # Create total power arrays
        if 'phi' not in rad_distribution:
            rad_distribution['phi'] = rD_phi
            rad_distribution['total_power'] = np.zeros(rD_phi.shape)
        else:
            # Checker to make sure that phi arrays match when doing helical distributions
            # (or more than one injection location)
            if not np.array_equal(rD_phi, rad_distribution['phi']):
                raise ValueError("Error! PHI arrays do not match in _post_process_calculations")
                
        # dphi already accounts for _rev1, _rev2, etc. for helical distributions
        dphi = Util_emis3D.find_dphi(rD_phi, mu, emissionName=emissionName)


        # Both directions share amplitude 'a'; their individual decay constant
        # 'b' controls how fast each falls off away from the injection location.
        a = params[f"a_{inj_loc_tag}"]
        b = params[f"b_{emissionName}_{inj_loc_tag}"]

        scale_ = Util_emis3D.scale_wrapper(
            a=a,
            b=b,
            phi=rD_phi,
            mu=mu,
            scale_def=scale_def,
            emissionName=emissionName,
            dphi=dphi,
        )

        # --- Now calculate the radiation around the vessel due to the radDist
        rD_power = rD_P_pol * scale_ * rD_scale
        
        rad_distribution[emissionName]["phi"] = np.asarray(rD_phi)
        rad_distribution[emissionName]["multiplication_factor"] = np.asarray(scale_)
        rad_distribution[emissionName]["total_power"] = np.asarray(rD_power)
        rad_distribution['total_power'] += rD_power

        
        tp = rad_distribution['total_power']
        tpf = np.max(tp) / (simpson(tp, x = rad_distribution['phi']) / (2.0 * np.pi))
        rad_distribution['toroidal_peaking_factor'] = tpf



    # CHECKER, plot the scale factors
    if True:
        plt.ion()
        f = plt.figure()
        ax = f.add_subplot(111)
        for emissionName in emissionNames:
            ax.plot(rad_distribution[emissionName]["phi"] , 
                    rad_distribution[emissionName]["total_power"],
                    label = emissionName)
        ax.plot(rad_distribution['phi'], rad_distribution['total_power'], linewidth = 2.0, color = 'black', label = 'Total Emission')
        ax.legend()
        ax.set_xlabel('phi [rad]')
        ax.set_ylabel('multiplication factor')
        plt.show()
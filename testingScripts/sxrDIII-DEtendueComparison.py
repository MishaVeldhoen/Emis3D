# sxrDIII-DEtendueComparison.py
"""
Calculates the etendue with Emis3D and compares it to the measured values
"""


# sxrChordTester.py
"""
Loads all of the bolometers on the diagnostic, calculates
the chord position based off the SXR input file(s) and overlays
the measured chords in the SXR input file(s).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np
from main.Tokamak import Tokamak

# import tokamaks.DIII_D.sxrSavers.Util_SXR as Util_SXR
from pathlib import Path

module_path = str(Path("../tokamaks/DIII-D/sxrSavers/").resolve())
sys.path.insert(0, module_path)
import Util_SXR  # type: ignore
from raysect.optical import AbsorbingSurface, NullMaterial  # type: ignore

t = Tokamak(
    tokamakName="DIII-D",
    mode="Build",
    reflections=False,
    loadBolometers=True,
)


etendue_ = {}
if t.info is not None:
    for b_ in t.info["Bolometer Groups"]:
        if b_ == "SX90PF":
            n_ = "SX90RP1F"
        elif b_ == "SX90MF":
            n_ = "SX90RM1F"
        elif b_ == "SX45F":
            n_ = "SXR45"
        else:
            n_ = b_
        etendue_[b_] = Util_SXR.get_calibration(160606, ArrayName=n_)


t.calc_etendues()

for b_ in t.bolometers:
    b_.calc_etendues()


plt.ion()


# --- Plot each individual bolometer
if t.info is not None:
    boloGroups = t.info["Bolometer Groups"]
    bolometers = t.bolometers

    num_figs = len(boloGroups)
    num_rows = int(np.ceil(num_figs / 4))  # no more than 4 across
    f = plt.figure(figsize=(15, 8))
    initilized = False

    # --- Loop over each bolometer group
    for ii, boloGroup in enumerate(boloGroups):
        f_ = f.add_subplot(num_rows, int(num_figs / num_rows), ii + 1)
        t._plot_first_wall(f_)
        t._plot_bolometers(
            f_, boloGroupName=boloGroup, plot_chord_info=True, plot_etendue=["SX45F07"]
        )

    plt.tight_layout()
    plt.show()

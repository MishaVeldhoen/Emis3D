# sxrChordTester.py
"""
Loads all of the bolometers on the diagnostic, calculates
the chord position based off the SXR input file(s) and overlays
the measured chords in the SXR input file(s).

TODO:
1. Some weird bug happens when I load more than one bolometer, it's like they don't
trace the rays to the wall correctly

"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np
from raysect.core import Point2D

from main.Tokamak import Tokamak


def point3d_to_rz(point):
    return Point2D(np.hypot(point.x, point.y), point.z)


t = Tokamak(
    tokamakName="JET",
    mode="Build",
    reflections=False,
    loadBolometers=True,
)


# Plot in 3D to see if it is at the correct toroidal location
if False:
    t.plot()

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
        t._plot_bolometers(f_, boloGroupName=boloGroup)

    plt.tight_layout()
    plt.show()

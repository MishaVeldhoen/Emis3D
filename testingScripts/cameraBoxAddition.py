# cameraBoxAddition.py
"""
Testing creting a box around a bolometer to have more accurate etendue calculations
"""


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np

from main.Tokamak import Tokamak
from raysect.core import Point3D
from main.Util import config_loader, point3d_to_rz
from raysect.primitive import Box
from raysect.optical import AbsorbingSurface  # type:ignore

t = Tokamak(
    tokamakName="DIII-D",
    mode="Build",
    reflections=False,
    loadBolometers=True,
)


bolo = t.bolometers[0].bolometer_camera
foil = bolo.foil_detectors[0]
slit = foil.slit


plt.ion()

"""
# Plot in 3D to see if it is at the correct toroidal location
if True:
    t.plot()
"""
# --- Plot each individual bolometer
if t.info is not None:
    boloGroups = t.info["Bolometer Groups"][0]
    f = plt.figure(figsize=(6, 8))
    f_ = f.add_subplot(1, 1, 1)
    t._plot_first_wall(f_)
    t._plot_bolometers(
        f_, boloGroupName=boloGroups, plot_chord_info=False, plot_etendue=["SX90PF09"]
    )

    plt.tight_layout()
    plt.show()

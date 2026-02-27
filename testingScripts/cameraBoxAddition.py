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

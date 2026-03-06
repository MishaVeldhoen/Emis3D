#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:43:28 2023

@author: bsteinlu

Updated and re-written during the refactor - JLH

"""

import os

import matplotlib.pyplot as plt
import numpy as np
from freeqdsk import geqdsk
from matplotlib import path
from raysect.core import Point3D
from raysect.core.math import rotate_z, translate
from raysect.optical import World
from raysect.optical.library import RoughTungsten
from raysect.optical import AbsorbingSurface, NullMaterial  # type: ignore

from raysect.primitive import Cylinder, import_stl, Subtract

from main.Diagnostic import Bolometer
from main.Fieldline_Tracer import Fieldline_Tracer
from main.Globals import *
from main.Util import (
    config_loader,
    point3d_to_rz,
    draw_radial_lines,
    find_intersection,
    rz_to_xyz,
    split_revolutions,
    compute_etendue_metric,
    get_rectangle_corners,
)


class Tokamak(object):

    def __init__(
        self,
        tokamakName=None,
        mode="Analysis",
        reflections=False,
        eqFileName=None,
        loadBolometers=False,
        verbose=False,
    ):
        """
        Basic tokamak class which loads information specific to the TokamakName.

        This class loads the configuration file for the tokamak within
        ../tokamaks/{DIII-D, JET, SPARC, etc}/{DIII-D, JET, SPARC, etc}_settings.yaml

        The file should contain information about the SXR/bolometer arrays,
        wall file location, volume, majorRadius, minorRadius, etc.

        INPUTS:

        tokamakName :: The name of the tokamak (e.g. DIII-D, SPARC)
        mode        :: Analysis or Build
        reflections :: Boolean, determines if the tokamak reflects the radiation
        eqFileName  :: The name of the equilibrium file to be used
        loadBolometers    :: Set to True to load the bolometers (needed to make radDists)
        """

        self.verbose = verbose
        if tokamakName not in SUPPORTED_TOKAMAKS:
            print(f"Please eneter a valid tokamak name!")
            print(f"Tokamaks currently supported are: {SUPPORTED_TOKAMAKS}")
            raise Exception

        else:
            # --- Load the configuration file
            self._load_config_file(tokamakName, mode, reflections, eqFileName)

            # --- Set the general input directory
            self.input_dir = os.path.join(
                EMIS3D_TOKMAK_DIRECTORY, tokamakName, "inputs"
            )

            # --- Run the startup program
            self._tokamak_startup(loadBolometers=loadBolometers)

    def _load_config_file(self, tokamakName, mode, reflections, eqFileName) -> None:
        """
        Loads the configuration file for the given tokamak
        """

        pathFileName = os.path.join(
            EMIS3D_TOKMAK_DIRECTORY, tokamakName, f"{tokamakName}_settings.yaml"
        )

        # --- Load the configuration file, if it exists
        if os.path.isfile(pathFileName):
            self.info = config_loader(pathFileName)
        else:
            print(
                f"Could not load the configuration file, file does not exist: {pathFileName}"
            )

        # --- Create the self.info dict if the file is not loaded
        if self.info == None:
            self.info = {}

        # Angle conventions used in each tokamak are different from that used in Cherab.
        # Emis3D uses the Cherab angle convention. This angle is subtracted in the evaluate
        # statements in RadDist to make the angles match.
        torConventionPhis = {"JET": np.pi / 2.0, "SPARC": 0.0, "DIII-D": 0.0}
        self.info["torConventionPhi"] = torConventionPhis[tokamakName]
        self.info["tokamakName"] = tokamakName
        self.info["mode"] = mode
        self.info["reflections"] = reflections
        self.info["eqFileName"] = eqFileName

    def _tokamak_startup(self, loadBolometers=False) -> None:
        """
        This definition will:
        1. Load the equilibrium file, if given
        2. Load the bolometers
        3. Load the wall file, defaults to what is in the equilibrium file
        4. Builds the tokamak (if mode = Build)
        """

        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        # --- Create the raysect world
        self.world = World()

        # --- Load the equilibrium file
        if self.info["eqFileName"] is not None:
            self._load_eqFile()

        # --- Load the bolometers
        if loadBolometers:
            self._load_bolometers()

        # --- Load the first wall
        self._load_first_wall()

        # --- Create boundries in the raysect world
        if self.info["mode"].upper() == "BUILD":
            self._build_tokamak()

    def _load_eqFile(self) -> None:
        """
        Definition loads the equilibrium file, then uses the wall information, if it is there
        """
        pathFileName = ""
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        try:
            pathFileName = os.path.join(
                EMIS3D_INPUTS_DIRECTORY,
                self.info["tokamakName"],
                "eqdsks",
                self.info["eqFileName"],
            )
            if os.path.isfile(pathFileName):
                with open(pathFileName) as f:
                    self.gfile = geqdsk.read(f)
                if self.verbose:
                    print(f"Loaded equilibrium file: {pathFileName}")
            else:
                print(f"Equilibrium file not found!")
        except Exception as e:
            print(f"Could not read the equlibrium file, error: {e}")
            print(f"Tried to read it here: {pathFileName}")

    def _load_first_wall(self) -> None:
        """
        Loads the first wall from the given text file. The file name should be within the
        tokamak setup file ["MACHINE"]["wallFileName"], and the file should be within the
        EMIS3D_TOKMAK_DIRECTORY/inputs/...txt
        """

        # --- Checkers
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        rzarray = None

        # --- Default to the one from the eqFile, if it exists
        if hasattr(self, "gfile"):
            if hasattr(self.gfile, "rlim") and hasattr(self.gfile, "zlim"):
                if self.gfile.rlim is not None and self.gfile.zlim is not None:
                    rzarray = np.vstack((self.gfile.rlim, self.gfile.zlim)).T
                else:
                    rzarray = None

        # --- Load the wall from the text file
        elif "wallFileName" in self.info["MACHINE"]:
            pathFileName = os.path.join(
                self.input_dir, self.info["MACHINE"]["wallFileName"]
            )
            try:
                rzarray = np.loadtxt(pathFileName, skiprows=0)
            except:
                rzarray = np.loadtxt(pathFileName, delimiter=",", skiprows=0)

        else:
            self.wall = None

        # --- Store the wall information
        if rzarray is not None:
            self.wall = {}
            self.wall["rzarray"] = np.array(rzarray)
            self.wall["minr"] = min(rzarray[:, 0])
            self.wall["maxr"] = max(rzarray[:, 0])
            self.wall["minz"] = min(rzarray[:, 1])
            self.wall["maxz"] = max(rzarray[:, 1])
            self.wall["wallcurve"] = path.Path(rzarray)

    def _build_tokamak(self, load_stl=False) -> None:
        """
        Builds the tokmak within the raysect world.
        """
        PFC_STL_PATH = ""

        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        # Building a closed universe
        if hasattr(self, "wall") and load_stl == False:
            if self.wall is not None:
                # --- Find the z limits
                if "Bolometer Z Limits" in self.info:
                    minz = np.min(
                        [self.wall["minz"], self.info["Bolometer Z Limits"][0]]
                    ).astype(float)
                    maxz = np.max(
                        [self.wall["maxz"], self.info["Bolometer Z Limits"][1]]
                    ).astype(float)
                else:
                    minz = self.wall["minz"]
                    maxz = self.wall["maxz"]
                height = np.abs(maxz) + np.abs(minz)
                offset = minz

                # --- Find the R limits
                if "Bolometer R Limits" in self.info:
                    minR = np.min(
                        [self.wall["minr"], self.info["Bolometer R Limits"][0]]
                    ).astype(float)
                    maxR = np.max(
                        [self.wall["maxr"], self.info["Bolometer R Limits"][1]]
                    ).astype(float)
                else:
                    minR = self.wall["minr"]
                    maxR = self.wall["maxr"]

                # --- Outer wall
                cylinder_outer = Cylinder(
                    radius=maxR + 0.2,
                    height=height + 0.2,
                    name="Outer wall",
                )

                # --- Inner wall
                # NOTE: This should be modified if any bolometers are inside of this radius,
                # otherwise Raysect will not trace the chords correctly
                cylinder_inner = Cylinder(
                    radius=minR - 0.2,
                    height=height + 0.2,
                    name="Inner wall",
                )

                wall = Subtract(
                    cylinder_outer,
                    cylinder_inner,
                    material=AbsorbingSurface(),  # Do NOT CHANGE THIS to a NullSurface, otherwise the foil.observe will not work properly
                    name="Tokamak Wall",
                    parent=self.world,
                    transform=translate(0, 0, offset - 0.1),
                )

                # --- Have the emission surface inside the tokamak
                # --- Outer wall
                emiss_outer = Cylinder(
                    radius=self.wall["maxr"] + 1.0,
                    height=height,
                    name="Outer wall",
                )

                # --- Inner wall
                emiss_inner = Cylinder(
                    radius=self.wall["minr"],
                    height=height,
                    name="Inner wall",
                )

                emiss_surface = Subtract(
                    emiss_outer,
                    emiss_inner,
                    name="Emission Surface",
                    parent=self.world,
                    transform=translate(0, 0, offset),
                    material=NullMaterial(),
                )

        # --- Load the CAD file
        elif load_stl:
            try:
                # --- Standard scale if the machine is in meters
                STL_SCALE = 1.0
                if self.info["MACHINE"]["STL_UNITS"].lower() == "mm":
                    STL_SCALE = 1.0e-3

                PFC_STL_PATH = os.path.join(
                    self.input_dir,
                    "CAD_stl_files",
                    self.info["MACHINE"]["PFC_STL_PATH"],
                )
                if os.path.isfile(PFC_STL_PATH):
                    pfcs = import_stl(PFC_STL_PATH, scaling=STL_SCALE)
                    # pfcs.transform=rotate_x(90)
                    pfcs.transform = rotate_z(60)  # for r_li first wall with cutouts
                    pfcs.material = RoughTungsten(0.6)
                    pfcs.name = "PFCs"
                    pfcs.parent = self.world

                    if self.info["reflections"] == False:
                        for child in self.world.children:
                            child.material = AbsorbingSurface()
            except Exception as e:
                print(f"Could not load the CAD tokamak file: {PFC_STL_PATH}")
                print(f"Error: {e}")

        else:
            print("Building the tokamak failed")
            print(
                f"Could not load the stl file for this tokamak, {PFC_STL_PATH} or the wall"
            )
            print("was not loaded")

    def _load_bolometers(self) -> None:
        """
        Loads the bolometers for the given tokamak.

        The files should be in the tokamak input folder,
        /tokamaks/DIII-D/inputs/sxrInfo/

        """
        self.bolometers = []

        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return
        # --- Store all of the bolometer groups, used to make plotting easier
        self.info["Bolometer Groups"] = []

        # --- Find the maximum and minimum locations of the bolometer, used to create the world
        r_limits = [1000, 0]
        z_limits = [1000, -1000]

        # --- Create each bolometer in the tokamak configuration file
        for bolo in self.info["BOLOMETERS"]:
            b_ = Bolometer(
                world=self.world,
                tokamakName=self.info["tokamakName"],
                configFileName=self.info["BOLOMETERS"][bolo]["configFileName"],
            )
            self.bolometers.append(b_)

            if b_.info is not None:
                offset = np.abs(b_.info["SLIT_SENSOR_SEPARATION"])
                if b_.info["CAMERA_POSITION_R_Z_PHI"][0] + offset > r_limits[1]:
                    r_limits[1] = b_.info["CAMERA_POSITION_R_Z_PHI"][0] + offset
                if b_.info["CAMERA_POSITION_R_Z_PHI"][0] - offset < r_limits[0]:
                    r_limits[0] = b_.info["CAMERA_POSITION_R_Z_PHI"][0] - offset
                if b_.info["CAMERA_POSITION_R_Z_PHI"][1] + offset > z_limits[1]:
                    z_limits[1] = b_.info["CAMERA_POSITION_R_Z_PHI"][1] + offset
                if b_.info["CAMERA_POSITION_R_Z_PHI"][1] - offset < z_limits[0]:
                    z_limits[0] = b_.info["CAMERA_POSITION_R_Z_PHI"][1] - offset

                if b_.info["GROUP_NAME"] not in self.info["Bolometer Groups"]:
                    self.info["Bolometer Groups"].append(b_.info["GROUP_NAME"])

        # --- Used to create the cherab bounding box for the bolometers
        self.info["Bolometer R Limits"] = r_limits
        self.info["Bolometer Z Limits"] = z_limits

    def _inside_tokamak(self, points) -> bool:
        """
        Checks to see if the np.column_stacked R, z points are within the tokamak wall
        """
        if self.wall is None:
            raise RuntimeError(
                "Tokamak wall is not initialized. Ensure that the wall attribute is set before calling inside_tokamak()."
            )

        wallcurve = self.wall["wallcurve"]

        return wallcurve.contains_points(points)

    def _make_raysect_surface_transparent(self, surfaceName="") -> None:
        """
        Makes the raysect object transparent. This is done after observation in order
        for the sightlines to trace properly
        """
        # --- Add the emitter to the tokamak wall
        for val in self.world.children:
            if val.name == surfaceName:
                val.material = NullMaterial()

    def _change_emission_surface_material(self, material) -> None:
        """
        Changes the material of the emission surface, used for testing purposes
        """
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        for child in self.world.children:
            if child.name == "Emission Surface":
                child.material = material

    def _plot_first_wall(self, ax=None) -> None:
        """
        Plots the first wall from self.wall["wallcurve"].

        Input: ax : A subplot of the plt.figure()
            f = plt.figure()
            ax = f.add_subplot(111)
        """
        if self.wall is None:
            print("No wall information loaded, cannot continue!")
            return

        show_plt = False
        if ax is None:
            f = plt.figure(figsize=(4, 7))
            ax = f.add_subplot(111)
            show_plt = True

        ax.plot(
            self.wall["wallcurve"].vertices[:, 0],
            self.wall["wallcurve"].vertices[:, 1],
            color="black",
            linewidth=2.0,
        )
        ax.set_xlabel("R [m]", fontsize=12)
        ax.set_ylabel("z [m]", fontsize=12)
        ax.set_aspect("equal")

        if show_plt:
            plt.show()

    def _plot_labels(self, ax) -> None:
        """
        Adds labels to a 3d plot, used for self.plot
        """

        if self.wall is None:
            print("No wall information loaded, cannot continue!")
            return

        # --- Generate points for the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x = self.wall["maxr"] * np.cos(theta)
        y = self.wall["maxr"] * np.sin(theta)
        z = np.zeros_like(theta)

        ax.plot(x, y, z, color="black")

        # --- Define the angles (in degrees) where labels and tick marks should be added
        angles_deg = [0, 90, 135, 180, 225, 270]
        tick_length = 0.1  # Length of each tick mark

        for angle in angles_deg:
            angle_rad = np.deg2rad(angle)  # Convert degrees to radians

            # Coordinates on the circle
            x_circle = self.wall["maxr"] * np.cos(angle_rad)
            y_circle = self.wall["maxr"] * np.sin(angle_rad)
            z_circle = 0  # Always on z = 0 plane

            # Compute the end point of the tick mark (extending outward from the circle)
            x_tick = self.wall["maxr"] * np.cos(angle_rad) * (1 + tick_length)
            y_tick = self.wall["maxr"] * np.sin(angle_rad) * (1 + tick_length)
            z_tick = 0  # Still on the z = 0 plane

            # Add the label at the circle
            ax.text(
                x_circle,
                y_circle,
                z_circle,
                f"{angle}°",
                fontsize=12,
                color="red",
                horizontalalignment="center",
                verticalalignment="center",
            )

            # Draw the tick mark as a short line segment from the circle to the tick endpoint
            ax.plot(
                [x_circle, x_tick],
                [y_circle, y_tick],
                [z_circle, z_tick],
                color="black",
                lw=2,
            )

    def _plot_channel_with_envelope(
        self, ax, foil, slit, length=None, debug=False
    ) -> None:

        if ax is not None:

            if debug:
                print("\n--- FOIL GEOMETRY ---")
                print(f"Width  : {foil.y_width:.6f} m")
                print(f"Height : {foil.x_width:.6f} m")

                print("\n--- SLIT GEOMETRY ---")
                print(f"Width  : {slit.dy:.6f} m")
                print(f"Height : {slit.dx:.6f} m")

                foil_center = foil.to_root() * foil.centre_point
                slit_center = slit.to_root() * slit.centre_point

                separation = np.sqrt(
                    (foil_center.x - slit_center.x) ** 2
                    + (foil_center.y - slit_center.y) ** 2
                    + (foil_center.z - slit_center.z) ** 2
                )

                print("\n--- FOIL to SLIT SEPARATION ---")
                print(f"Center-to-center distance: {separation:.6f} m")

            # Get corners
            foil_corners = get_rectangle_corners(foil)
            slit_corners = get_rectangle_corners(slit)

            # Identify upper/lower in Z
            foil_sorted = sorted(foil_corners, key=lambda p: p.z)
            slit_sorted = sorted(slit_corners, key=lambda p: p.z)

            foil_lower = foil_sorted[0]
            foil_upper = foil_sorted[-1]
            slit_lower = slit_sorted[0]
            slit_upper = slit_sorted[-1]

            # Two diagonal options
            option1 = [(foil_lower, slit_upper), (foil_upper, slit_lower)]

            option2 = [(foil_lower, slit_lower), (foil_upper, slit_upper)]

            # Compute metric
            metric1 = sum(compute_etendue_metric(a, b) for a, b in option1)
            metric2 = sum(compute_etendue_metric(a, b) for a, b in option2)

            if metric1 >= metric2:
                chosen = option1
            else:
                chosen = option2

            # Plot foil & slit corners
            for p in foil_corners:
                r, z = point3d_to_rz(p)
                ax.scatter(r, z)

            for p in slit_corners:
                r, z = point3d_to_rz(p)
                ax.scatter(r, z)

            # Plot envelope lines
            for a, b in chosen:
                start = np.array([a.x, a.y, a.z])
                end = np.array([b.x, b.y, b.z])

                direction = end - start
                distance = np.linalg.norm(direction)
                direction /= distance

                if length is not None:
                    new_end = start + direction * length
                else:
                    new_end = end

                r1, z1 = point3d_to_rz(a)

                temp_point = a.__class__(*new_end)
                r2, z2 = point3d_to_rz(temp_point)

                plt.plot([r1, r2], [z1, z2], linewidth=2, color="tab:red")

    def _plot_bolometers(
        self, ax, boloGroupName, plot_chord_info=False, plot_etendue=[]
    ) -> None:
        """
        Plots the chords for a specific bolometer group
        ax :: matplotlib.plot
        boloGroupName :: The GROUP_NAME in each bolometer file
        plot_chord_info :: Plot r0, rf, etc. in each bolometer file, typically used for initial
                        debugging of new bolometers since it compares Cherab to known chord positions
        """

        # --- Change the inner wall to an absorbing surface, so the chords have something intersect with
        self._change_emission_surface_material(AbsorbingSurface())

        # --- Make sure self.info is initiated
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        # --- Check to see if boloGroupName is in self.info['Bolometer Groups']
        if boloGroupName not in self.info["Bolometer Groups"]:
            print(f"{boloGroupName} is not found in self.info['Bolometer Groups']")
            return

        # --- Loop over each bolometer, only plot those with the correct group name
        label_cherab = True
        for bolo in self.bolometers:
            if bolo.info["GROUP_NAME"] == boloGroupName:

                # --- Over plot the chords in the cofig file
                if plot_chord_info and bolo.info is not None:
                    if "r0" in bolo.info:
                        label_ = "Chords from config file"

                        for ii in range(len(bolo.info["r0"])):
                            # --- Check to see if the currnt plot already has the correct label
                            _, labels = ax.get_legend_handles_labels()
                            if "Chords from config file" in labels:
                                label_ = "__no_legend__"

                            ax.plot(
                                [bolo.info["r0"][ii], bolo.info["rf"][ii]],
                                [bolo.info["z0"][ii], bolo.info["zf"][ii]],
                                linewidth=2.0,
                                color="green",
                                label=label_,
                            )

                for foil in bolo.bolometer_camera.foil_detectors:

                    label = "__no_legend__"
                    if label_cherab:
                        label = "Chords from Raysect"
                        label_cherab = False

                    # --- Slit center
                    slit_rz = point3d_to_rz(foil.slit.centre_point)
                    ax.plot(slit_rz[0], slit_rz[1], "ko")

                    # --- Foil center
                    centre_rz = point3d_to_rz(foil.centre_point)
                    ax.plot(centre_rz[0], centre_rz[1], "kx")

                    # --- Ray-traced sightline
                    origin, hit, _ = foil.trace_sightline()
                    origin_rz = point3d_to_rz(origin)

                    if origin is not None and hit is not None:
                        origin_rz = point3d_to_rz(origin)
                        hit_rz = point3d_to_rz(hit)

                        ax.plot(
                            [origin_rz[0], hit_rz[0]],
                            [origin_rz[1], hit_rz[1]],
                            color="tab:blue",
                            linewidth=1.0,
                            label=label,
                        )

                        # --- Add the channel number
                        ch = ""
                        try:
                            if int(foil.name[-2:]):
                                ch = int(foil.name[-2:])
                        # For JET foils
                        except Exception:
                            n = foil.name.split("_")[1]
                            ch = n[2:]

                        ax.text(
                            hit_rz[0],
                            hit_rz[1],
                            ch,
                            fontsize="10",
                            ha="center",
                            va="center",
                            weight="bold",
                        )

                        # --- Highlight the etendue of each channel
                        if foil.name in plot_etendue:
                            dR = np.abs(origin_rz[0] - hit_rz[0])
                            dz = np.abs(origin_rz[1] - hit_rz[1])
                            length = np.sqrt(dR**2 + dz**2)

                            self._plot_channel_with_envelope(
                                ax, foil=foil, slit=foil.slit, length=length
                            )

                    # Debug arrow in ray direction
                    slit_rz = point3d_to_rz(foil.slit.centre_point)
                    direction = np.array(
                        [slit_rz[0] - origin_rz[0], slit_rz[1] - origin_rz[1]]
                    )
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                        scale = 0.05
                        ax.quiver(
                            origin_rz[0],
                            origin_rz[1],
                            direction[0] * scale,
                            direction[1] * scale,
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            color="red",
                        )

        ax.legend(loc="upper right")
        ax.set_title(boloGroupName)

        # --- Change the inner wall back so the etendue's can be calculated correctly
        self._change_emission_surface_material(NullMaterial())

    def get_ave_bolometer_tor_loc(self, boloGroupName):
        """
        Returns the average toroidal angle of the bolometer group
        """

        phi = []

        # --- Make sure self.info is initiated
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return None

        # --- Check to see if boloGroupName is in self.info['Bolometer Groups']
        if boloGroupName not in self.info["Bolometer Groups"]:
            print(f"{boloGroupName} is not found in self.info['Bolometer Groups']")
            return None

        # --- Loop over each bolometer, only plot add with the correct group name
        for bolo in self.bolometers:
            if bolo.info["GROUP_NAME"] == boloGroupName:
                phi.append(bolo.info["CAMERA_POSITION_R_Z_PHI"][2])

        return np.mean(np.array(phi))

    def plot(self, fieldLineStartPhi=None) -> None:
        """
        Plot the tokamak configuration in 3D

        Inputs:
            fieldLineStartPhi :: float
                                 field line start location in degrees
        """
        if self.wall is None:
            print("No wall information loaded, cannot continue!")
            return
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        self._plot_labels(ax)

        r = self.wall["wallcurve"].vertices[:, 0]
        z = self.wall["wallcurve"].vertices[:, 1]
        r = np.concatenate((r, [r[0]]))
        z = np.concatenate((z, [z[0]]))
        y = np.squeeze(np.zeros((1, len(r))))
        ax.plot(r, y, z, "black")
        ax.plot(r * (-1), y, z, "black")
        ax.plot(y, r, z, "black")
        ax.plot(y * (-1), r * (-1), z, "black")

        if hasattr(self, "bolometers"):

            if self.info["mode"] != "Build":
                print(
                    f"Building the tokamak! We need this to trace the chords for the bolometers"
                )
                self._build_tokamak()
            try:
                for bolo in self.bolometers:
                    for foil in bolo.bolometer_camera:
                        slit_centre = foil.slit.centre_point
                        ax.plot(slit_centre[0], slit_centre[1], slit_centre[2], "ko")
                        origin, hit, _ = foil.trace_sightline()
                        ax.plot(
                            foil.centre_point[0],
                            foil.centre_point[1],
                            foil.centre_point[2],
                            "kx",
                        )
                        ax.plot(
                            [origin[0], hit[0]],
                            [origin[1], hit[1]],
                            [origin[2], hit[2]],
                            "k",
                        )
                        ax.text(hit[0], hit[1], hit[2], foil.name)
            except Exception as e:
                print(f"Could not plot the bolometer chords, error: {e}")
                print(
                    f"Ensure that the tokamak is built before plotting the bolometers, currently in mode: {self.info['mode']}"
                )

        # --- Plot the given field line
        if hasattr(self, "fieldLines"):
            colors = ["red", "green", "blue", "orange", "purple", "brown"]
            if str(fieldLineStartPhi) in self.get_fieldLines_startPhis():
                for ii, dir_ in enumerate(
                    self.fieldLines[f"{fieldLineStartPhi}"]["directionNames"]
                ):
                    line_ = self.fieldLines[f"{fieldLineStartPhi}"][dir_]
                    ax.plot(
                        line_["x"],
                        line_["y"],
                        line_["z"],
                        color=colors[ii],
                        label=dir_,
                        linewidth=2.0,
                    )

            else:
                print(f"Input fieldLinePhi of {fieldLineStartPhi}, not availble!")
                print(f"Possible fieldLinePhi(s): {self.get_fieldLines_startPhis()}")

            ax.legend()

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")  # pyright: ignore[reportAttributeAccessIssue]
        ax.set_xlim(float(-2.5), float(2.5))
        ax.set_ylim(float(-2.5), float(2.5))
        ax.set_zlim(float(-2.5), float(2.5))  # type: ignore

        plt.show()

    def set_fieldlines(
        self, startR=[], startZ=[], startPhi=0.0, numTransists=1.0
    ) -> None:
        """
        Calculate the field line progress clockwise and counterclockwise from
        StartPhi. This supports inputs of multiple R and z locations.

        Input:
        startR      : Array of R points to start the field line
        startZ      : Array of z points to start the field line
        startPhi    : Phi start location of this field line in radians

        Stores:
        self.fieldLines[startPhi] : {}
        self.fieldLines[startPhi]['R']      : 2D array of R points for each phi progression [R, phi]
        self.fieldLines[startPhi]['z']      : 2D array of z points for each phi progression [z, phi]
        self.fieldLines[startPhi]['phi']    : 1D array of phi points
        self.fieldLines[startPhi]['L']      : 1D array of the distance to this phi location [m]
        self.fieldLInes[startPhi][startR, startZ, startPhi] : stores the intial data points

        This is done so you can get an array of all of the R and z points at a particular phi location,
        this vectorized format should be faster than storing them in a different manner.
        """

        startPhideg = f"{int(np.rad2deg(startPhi))}"
        # --- Initilize the arrays
        if not hasattr(self, "fieldLines"):
            self.fieldLines = {}
        self.fieldLines[startPhideg] = {}
        self.fieldLines[startPhideg]["NumTransists"] = numTransists
        self.fieldLines[startPhideg]["directionNames"] = []
        self.fieldLines[startPhideg]["startR"] = startR
        self.fieldLines[startPhideg]["startZ"] = startZ
        self.fieldLines[startPhideg]["startPhi"] = startPhi

        # --- Loop over the direction
        direction_prefix_ = ["counterClock", "clockwise"]
        numTrans = [numTransists, (-1.0) * numTransists]
        for ii, direction_prefix in enumerate(direction_prefix_):
            # --- Loop over the R, z coordinates, store the result
            for jj in range(len(startR)):
                if self.verbose:
                    print(
                        f"Calculating fields in the {direction_prefix} direction from\tR={startR[jj]:.2f}m, z={startZ[jj]:.2f}m"
                    )
                tracer = Fieldline_Tracer(
                    StartR=startR[jj],
                    StartZ=startZ[jj],
                    StartPhi=startPhi,
                    gfile=self.gfile,
                    NumTor=500,
                )
                # --- Trace the field line in the given direction
                tracer.trace(NumTransits=numTrans[ii])

                # --- Find the components for each revolution
                d_ = tracer.data
                rev = split_revolutions(
                    d_["x"], d_["y"], d_["z"], d_["phi"], d_["R"], d_["L"]
                )

                # --- Initilize the arrays
                for kk in range(int(numTransists)):
                    direction = f"{direction_prefix}_rev{kk}"
                    self.fieldLines[startPhideg]["directionNames"].append(direction)
                    if direction not in self.fieldLines[startPhideg]:
                        self.fieldLines[startPhideg][direction] = {}
                        for val in ["R", "L", "x", "y", "z"]:
                            self.fieldLines[startPhideg][direction][val] = np.zeros(
                                (len(startR), len(rev[kk]["phi"]))
                            )

                    # --- Store the data
                    for val in ["R", "L", "x", "y", "z"]:
                        self.fieldLines[startPhideg][direction][val][jj, :] = rev[kk][
                            val
                        ].flatten()
                    # --- Phi should be the same for each one of them, so we only need to store it once
                    if jj == 0:
                        self.fieldLines[startPhideg][direction]["phi"] = rev[kk]["phi"]

    def get_fieldLines_startPhis(self) -> list:
        return list(self.fieldLines.keys())

    def find_RZ_Fline(self, startPhi, emissionName, inputPhis=[]):
        """
        Returns the R and z arrays for the given input inputPhi location.

        Phi should always be positive!

        Method based on Divakr's answer here, this should be faster than the simple np.abs(x - x0).argmin():
        https://stackoverflow.com/questions/45349561/find-nearest-indices-for-one-array-against-all-values-in-another-array-python
        """

        B = np.array(self.fieldLines[startPhi][emissionName]["phi"])
        A = np.array(inputPhis)
        L = np.array(B).size
        sidx_B = B.argsort()
        sorted_B = B[sidx_B]
        sorted_idx = np.searchsorted(sorted_B, A)
        sorted_idx[sorted_idx == L] = L - 1
        mask = (sorted_idx > 0) & (
            (np.abs(A - sorted_B[sorted_idx - 1]) < np.abs(A - sorted_B[sorted_idx]))
        )
        flInd = sidx_B[sorted_idx - mask]
        R = self.fieldLines[startPhi][emissionName]["R"][:, flInd]
        z = self.fieldLines[startPhi][emissionName]["z"][:, flInd]
        return R, z

    def create_cameras(self, dtheta=10, dtheta_camera=4.9, phi=0):
        """
        Creates cherab cameras around the vacuum vessel.
        Currently works for DIII-D, need to test other tokamaks
        TODO:
        1. Remove R = 1.5, this is the length of the "spoke" since we use a spoke
        pattern from the center of the vessel to equally space the cameras around the vessel.
        This should just be the minor radius + some
        """

        # Plot first wall curve
        tok_r, tok_z = [0], [0]
        if self.wall is not None:
            r = self.wall["wallcurve"].vertices[:, 0]
            z = self.wall["wallcurve"].vertices[:, 1]
        r0 = (np.max(tok_r) - np.min(tok_r)) / 2.0 + np.min(tok_r)
        z0 = (np.max(tok_z) - np.min(tok_z)) / 2.0

        MACHINE_AXIS_3D = Point3D(r0, 0.0, z0)

        # --- Create the spoke pattern for the cameras, the upper and lower are the camera's width
        lines = {}
        lines_upper = {}
        lines_lower = {}

        for angle in np.arange(0, 360, dtheta):
            lines[angle] = draw_radial_lines(r0, z0, R=1.5, angle_deg=float(angle))

            lines_upper[angle] = draw_radial_lines(
                r0, z0, R=1.5, angle_deg=float(angle), initial_offset=dtheta_camera
            )
            lines_upper[angle]["offset"] = dtheta_camera
            lines_lower[angle] = draw_radial_lines(
                r0, z0, R=1.5, angle_deg=float(angle), initial_offset=-dtheta_camera
            )
            lines_lower[angle]["offset"] = -dtheta_camera

        # --- Fit each segement to a higher resolution, depending on the distance between the two segements
        r_ = []
        z_ = []
        for ii in range(len(tok_r)):
            if ii == len(tok_r) - 1:
                loc_ = 0
            else:
                loc_ = ii + 1
            r_.append(tok_r[ii])
            z_.append(tok_z[ii])

            # --- Length of this segment
            ds = np.sqrt(
                (tok_r[loc_] - tok_r[ii]) ** 2 + (tok_z[loc_] - tok_z[ii]) ** 2
            )

            if ds > 1.0:
                npts = 40
            else:
                npts = 20
            if ds > 0.04:
                # --- Fit the points
                if tok_r[loc_] == tok_r[ii]:
                    nx = [tok_r[loc_]] * npts
                else:
                    nx = np.linspace(tok_r[loc_], tok_r[ii], npts)
                ny = np.poly1d(
                    np.polyfit([tok_r[loc_], tok_r[ii]], [tok_z[loc_], tok_z[ii]], 1)
                )(nx)

                for x_ in nx:
                    r_.append(x_)
                for y_ in ny:
                    z_.append(y_)
        tok_r = r_.copy()
        tok_z = z_.copy()

        # --- Find where the center of camera intersects the vessel
        self.cameras = {}
        for ii, line in enumerate(lines):
            self.cameras[ii] = {}
            self.cameras[ii]["theta"] = line
            # r, z = Util_D3D.find_intersection(lines[line], tok_r, tok_z)
            # cameras[ii]["detector_center"] = Point3D(r, 0.0, z)
            r, z = find_intersection(lines_upper[line], tok_r, tok_z)
            x, y, z = rz_to_xyz(r, z, phi)
            self.cameras[ii]["p1"] = Point3D(x, y, z)

            r, z = find_intersection(lines_lower[line], tok_r, tok_z)
            x, y, z = rz_to_xyz(r, z, phi)
            self.cameras[ii]["p2"] = Point3D(x, y, z)

            self.cameras[ii]["y_vector_full"] = self.cameras[ii]["p1"].vector_to(
                self.cameras[ii]["p2"]
            )
            self.cameras[ii]["y_vector"] = self.cameras[ii]["y_vector_full"].normalise()
            self.cameras[ii]["y_width"] = self.cameras[ii]["y_vector_full"].length
            self.cameras[ii]["detector_center"] = (
                self.cameras[ii]["p1"] + self.cameras[ii]["y_vector_full"] * 0.5
            )

            x, y, z = rz_to_xyz(MACHINE_AXIS_3D[0], MACHINE_AXIS_3D[1], phi)
            self.cameras[ii]["normal_vector"] = (
                self.cameras[ii]["detector_center"].vector_to(Point3D(x, y, z))
            ).normalise()  # inward pointing

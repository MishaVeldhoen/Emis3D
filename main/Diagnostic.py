#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:43:28 2023

@author: bsteinlu

Re-wrote this during the refactor to only include a bolometer class. All of the
physical information is now stored in a self.info dictionary instead of doing
self.ax0, etc. for each variable. -JLH August, 2025.

TODO
1. Add more camera input options (from stl files, for example)
"""


import os

import numpy as np
from cherab.tools.observers import BolometerCamera, BolometerFoil, BolometerSlit
from raysect.core import (
    AffineMatrix3D,
    Node,
    Point3D,
    Vector3D,
    translate,
)

from main.Globals import *
from main.Util import RPhi_To_XY, config_loader, rotate_vector


class Bolometer(object):
    """
    Basic bolometer class loads the configuration file and creates the bolometer.
    The bolometer configuration file names should be defined in the tokamak
    settings file and each file should be in the tokamaks/x/inputs/sxrInfo/ folder.
    """

    def __init__(
        self,
        world=None,
        tokamakName=None,
        configFileName=None,
    ):

        self.world = world
        self.configFileName = configFileName
        self._load_config_file(tokamakName=tokamakName)
        self._build()

    def _load_config_file(self, tokamakName) -> None:
        """
        Loads the configuration file for the given diagnostic
        """

        pathFileName = join(
            EMIS3D_TOKMAK_DIRECTORY,
            tokamakName,
            "inputs",
            "sxrInfo",
            f"{self.configFileName}",
        )

        # --- Load the configuration file, if it exists
        if os.path.isfile(pathFileName):
            self.info = config_loader(pathFileName)
            if self.info is not None and "NAME" in self.info:
                self.name = self.info["NAME"]
            else:
                self.name = None
                print(
                    f"Configuration file loaded but missing 'NAME' key: {pathFileName}"
                )
        else:
            self.info = None
            self.name = None
            print(
                f"Could not load the configuration file, file does not exist: {pathFileName}"
            )

        # --- Load SubArray configuration files
        if self.info is not None and "SUBARRAY_NAMES" in self.info:
            if len(self.info["SUBARRAY_NAMES"]) > 0:
                for ii, fileName in enumerate(self.info["SUBARRAY_CONFIG_NAMES"]):
                    pathFileName = join(
                        EMIS3D_TOKMAK_DIRECTORY,
                        tokamakName,
                        "inputs",
                        "sxrInfo",
                        f"{fileName}",
                    )
                    if "SUBARRAYS" not in self.info:
                        self.info["SUBARRAYS"] = {}
                    self.info["SUBARRAYS"][self.info["SUBARRAY_NAMES"][ii]] = (
                        config_loader(pathFileName)
                    )

    def _build(self) -> None:
        """
        Definition to build the camera. This will call the correct definition based
        on the data within the bolometer configuration file
        """
        if self.info is not None and "BUILD_TYPE" in self.info:
            if self.info["BUILD_TYPE"] == "FROM PRIMITIVES":
                self._build_from_primitives()
            elif self.info["BUILD_TYPE"] == "KB5V":
                self.load_kb5_camera(camera_id="KB5V", parent=self.world)
            else:
                print(f"Build type of {self.info['BUILD_TYPE']} not yet supported!")
        else:
            print(
                "Bolometer configuration info is missing or incomplete; cannot build camera."
            )

    def _build_from_primitives(self) -> None:
        """
        Builds the bolometer based off the example found here:
        https://www.cherab.info/demonstrations/bolometry/camera_from_primitives.html#bolometer-from-primitives

        Notes:
        If two arrays are close together, it is best to include them within the same
        "bounding box," otherwise some of the raytraced chords might intersect the neighboring
        box

        This will be an array of channels corresponding to the NUM_CHANNELS
        and CHANNEL_TAGS parameters within the sxr configuration file

        The transform command should probably be split off into its own definition, since
        it should be universal
        """

        if self.info is None:
            print(
                "Bolometer configuration info is missing; cannot build from primitives."
            )
            return

        # --- Convenient constants
        XAXIS = Vector3D(1, 0, 0)
        YAXIS = Vector3D(0, 1, 0)
        ZAXIS = Vector3D(0, 0, 1)
        ORIGIN = Point3D(0, 0, 0)

        # --- Constants from the configuration file
        SLIT_WIDTH = self.info["SLIT_WIDTH"]
        SLIT_HEIGHT = self.info["SLIT_HEIGHT"]
        FOIL_WIDTH = self.info["FOIL_WIDTH"]
        FOIL_HEIGHT = self.info["FOIL_HEIGHT"]
        FOIL_CORNER_CURVATURE = self.info["FOIL_CORNER_CURVATURE"]
        SLIT_SENSOR_SEPARATION = self.info["SLIT_SENSOR_SEPARATION"]
        FOIL_SEPARATION = self.info["FOIL_SEPARATION"]
        CAMERA_POSITION_R_Z_PHI = self.info["CAMERA_POSITION_R_Z_PHI"]
        CAMERA_PHI_RAD = np.deg2rad(CAMERA_POSITION_R_Z_PHI[2])
        x, y = RPhi_To_XY(CAMERA_POSITION_R_Z_PHI[0], CAMERA_PHI_RAD)
        CAMERA_POSITION_X_Y_Z = (x, y, CAMERA_POSITION_R_Z_PHI[1])

        # --- Instance of the bolometer camera, set camera_geometry to None, add it later
        bolometer_camera = BolometerCamera(
            camera_geometry=None, parent=self.world, name=self.info["NAME"]
        )

        # --- Create a slit at the camera origin
        slit = BolometerSlit(
            slit_id=f"{self.info['NAME']} slit",
            centre_point=ORIGIN,
            basis_x=XAXIS,
            dx=SLIT_WIDTH,
            basis_y=YAXIS,
            dy=SLIT_HEIGHT,
            parent=bolometer_camera,
        )

        sensor = Node(
            name=f"{self.info['NAME']} sensor",
            parent=bolometer_camera,
            transform=translate(0, 0, -SLIT_SENSOR_SEPARATION),
        )

        # --- Create the foils relative to the sensor
        for ii, shift in enumerate(self.info["FOIL_POSITIONS"]):
            foil_transform = translate(shift * FOIL_SEPARATION, 0, 0) * sensor.transform
            foil = BolometerFoil(
                detector_id=self.info["CHANNEL_TAGS"][ii],
                centre_point=ORIGIN.transform(foil_transform),
                basis_x=XAXIS.transform(foil_transform),
                dx=FOIL_WIDTH,
                basis_y=YAXIS.transform(foil_transform),
                dy=FOIL_HEIGHT,
                slit=slit,
                parent=bolometer_camera,
                accumulate=False,
                curvature_radius=FOIL_CORNER_CURVATURE,
            )

            bolometer_camera.add_foil_detector(foil)

        # --- Translate the camera to the correct position
        origin_xyz = Vector3D(*CAMERA_POSITION_X_Y_Z)

        # --- Tilt the camera downward if it is downward facing
        sign = -1 if self.info["CAMERA_DOWNWARD_FACING"] else 1
        tilt_rad = sign * np.deg2rad(self.info["CAMERA_ROTATION"])
        skew_rad = np.deg2rad(self.info["SKEW_ANGLE"])

        e_R = Vector3D(np.cos(CAMERA_PHI_RAD), np.sin(CAMERA_PHI_RAD), 0).normalise()
        e_z = ZAXIS
        e_phi = e_z.cross(e_R).normalise()

        # Rotate e_r and e_phi about e_z for toroidal skew
        # e_R_skewed = rotate_vector(e_R, e_z, skew_rad).normalise() # this one not used
        e_phi_skewed = rotate_vector(e_phi, e_z, skew_rad).normalise()

        # Rotate e_z in R–z plane for poloidal tilt
        view_dir = rotate_vector(e_z, e_phi_skewed, tilt_rad).normalise()

        # Build orthonormal basis
        z_axis = view_dir  # Camera z-axis: the direction the camera "looks"
        y_axis = e_phi  # Camera y-axis: e_phi (toroidal direction)
        # Camera x-axis: y × z to form right-handed coordinate system
        x_axis = y_axis.cross(z_axis).normalise()

        # x_axis = y_axis.cross(z_axis).normalise()
        # --- Force orthogonality for safety
        y_axis = z_axis.cross(x_axis).normalise()

        # Build full rotation matrix from orthonormal basis
        basis_matrix = AffineMatrix3D(
            [
                x_axis.x,
                y_axis.x,
                z_axis.x,
                origin_xyz.x,
                x_axis.y,
                y_axis.y,
                z_axis.y,
                origin_xyz.y,
                x_axis.z,
                y_axis.z,
                z_axis.z,
                origin_xyz.z,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

        # Construct transform matrix
        bolometer_camera.transform = basis_matrix

        self.bolometer_camera = bolometer_camera

    def load_kb5_camera(self, camera_id, parent=None) -> None:
        """
        Loads the KB5 camera configuration from the locally stored csv file. The csv file can
        be found here: https://github.com/cherab/jet/tree/master/cherab/jet/bolometry/kb5

        Add DATA_PATH to the bolometer configuration file to specify the location of the csv files.

        Also, change the BUILD_TYPE to "KB5V" in the configuration file
        """

        if not self.info:
            raise ValueError("No info loaded for KB5 camera.")

        if "DATA_PATH" not in self.info:
            raise ValueError("No DATA_PATH in info for KB5 camera.")

        _DATA_PATH = self.info["DATA_PATH"]

        if camera_id == "KB5V":
            foils = np.loadtxt(
                os.path.join(_DATA_PATH, "kb5v_foils.csv"), delimiter=","
            )
            slits = np.loadtxt(
                os.path.join(_DATA_PATH, "kb5v_slits.csv"), delimiter=","
            )
        elif camera_id == "KB5H":
            foils = np.loadtxt(
                os.path.join(_DATA_PATH, "kb5h_foils.csv"), delimiter=","
            )
            slits = np.loadtxt(
                os.path.join(_DATA_PATH, "kb5h_slits.csv"), delimiter=","
            )
        else:
            raise ValueError("Unrecognised bolometer camera_id '{}'.".format(camera_id))

        num_slits = slits.shape[0]
        num_foils = foils.shape[0]

        bolometer_camera = BolometerCamera(name=camera_id, parent=parent)

        slit_objects = {}
        for i in range(num_slits):
            slit_data = slits[i]
            slit_id = "{}_Slit_#{}".format(camera_id, str(int(slit_data[0])))
            p1 = Point3D(slit_data[1], slit_data[2], slit_data[3])
            p2 = Point3D(slit_data[4], slit_data[5], slit_data[6])
            p3 = Point3D(slit_data[7], slit_data[8], slit_data[9])
            p4 = Point3D(slit_data[10], slit_data[11], slit_data[12])
            basis_x = p1.vector_to(p2).normalise()
            dx = p1.distance_to(p2)
            basis_y = p2.vector_to(p3).normalise()
            dy = p2.distance_to(p3)
            centre_point = Point3D(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4,
            )
            slit_objects[slit_id] = BolometerSlit(
                slit_id, centre_point, basis_x, dx, basis_y, dy, parent=bolometer_camera
            )

        for i in range(num_foils):
            foil_data = foils[i]
            foil_id = "{}_CH{}_Foil".format(camera_id, str(int(foil_data[0])))
            slit_id = "{}_Slit_#{}".format(camera_id, str(int(foil_data[1])))

            p1 = Point3D(foil_data[2], foil_data[3], foil_data[4])
            p2 = Point3D(foil_data[5], foil_data[6], foil_data[7])
            p3 = Point3D(foil_data[8], foil_data[9], foil_data[10])
            p4 = Point3D(foil_data[11], foil_data[12], foil_data[13])
            basis_x = p2.vector_to(
                p1
            ).normalise()  # switching orientation to ensure face orientation is correct
            dx = p1.distance_to(p2)
            basis_y = p2.vector_to(p3).normalise()
            dy = p2.distance_to(p3)
            centre_point = Point3D(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4,
            )

            # Shift backwards 3mm for all foils except those explicitly measured on back plate
            if camera_id == "KB5V":
                if i not in (9 - 1, 25 - 1, 32 - 1):
                    basis_z = basis_x.cross(basis_y).normalise()
                    centre_point = centre_point - basis_z * 0.0032
                # if i == 9 - 1:
                #     centre_point = centre_point + basis_x * 0.001
                if i == 31 - 1:
                    centre_point = centre_point - basis_x * 0.001
            else:
                basis_z = basis_x.cross(basis_y).normalise()
                centre_point = centre_point - basis_z * 0.0032

            foil = BolometerFoil(
                foil_id,
                centre_point,
                basis_x,
                dx,
                basis_y,
                dy,
                slit_objects[slit_id],
                parent=bolometer_camera,
            )

            bolometer_camera.add_foil_detector(foil)

        self.bolometer_camera = bolometer_camera

    def calc_etendues(self) -> None:
        """
        Calculate the etendue based on the geometery of the camera.
        Taken from the Cherab demo:
        https://www.cherab.info/demonstrations/bolometry/calculate_etendue.html

        Raytracing is not used here, since we did not load any CAD files for the bolometer.

        Can add it in the future, so I just left the commented-out code here.
        """

        self.raytraced_etendues = []
        self.raytraced_errors = []
        analytic_etendues = []
        for foil in self.bolometer_camera:
            raytraced_etendue, raytraced_error = foil.calculate_etendue(
                ray_count=100_000
            )
            Adet = foil.x_width * foil.y_width
            Aslit = foil.slit.dx * foil.slit.dy
            costhetadet = foil.sightline_vector.normalise().dot(foil.normal_vector)
            costhetaslit = foil.sightline_vector.normalise().dot(
                foil.slit.normal_vector
            )
            distance = foil.centre_point.vector_to(foil.slit.centre_point).length
            analytic_etendue = Adet * Aslit * costhetadet * costhetaslit / distance**2
            print(
                "{} raytraced etendue: {:.4g} +- {:.1g} analytic: {:.4g}".format(
                    foil.name, raytraced_etendue, raytraced_error, analytic_etendue
                )
            )
            self.raytraced_etendues.append(raytraced_etendue)
            self.raytraced_errors.append(raytraced_error)
            analytic_etendues.append(analytic_etendue)
        self.etendues = analytic_etendues
        self.etendues_error = np.array(analytic_etendues) * 0.1

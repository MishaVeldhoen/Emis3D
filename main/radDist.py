# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:12:06 2021

@author: bemst

Re-organized pretty much everything, added parallelization, and a lot more during the refactor -JLH Aug., 2025

Added new Helical class that expands with the field lines, moved old helical class to helicalRing -JLH Mar. 2026

TODO: Update powerPerBin and uncomment it from the self.build()

"""

import os

import numpy as np
from cherab.tools.emitters import RadiationFunction
from matplotlib import cm
from raysect.optical import VolumeTransform  # type: ignore

import main.Util_radDist as Util_radDist
from main.Globals import *
from main.Tokamak import Tokamak
from main.Util import XY_To_RPhi, convert_arrays_to_list, save_json
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class RadDist(ABC):
    """
    Parent RadDist class.
    """

    def __init__(self, startR=2.96, startZ=0.0, config={}):

        self.info = config
        self.info["startR"] = startR
        self.info["startZ"] = startZ
        if "injectionLocation" in config:
            self.info["startPhi"] = config["injectionLocation"]
            self.info["startPhiRad"] = np.deg2rad(config["injectionLocation"])

    def _build_tokamak(
        self,
        tokamakName="",
        mode="Build",
        reflections=False,
        eqFileName=None,
        loadBolometers=True,
    ) -> None:
        """
        Initializes an instance of the tokamak class
        """
        self.tokamak = Tokamak(
            tokamakName=tokamakName,
            mode=mode,
            reflections=reflections,
            eqFileName=eqFileName,
            loadBolometers=loadBolometers,
        )

    def _evalulateCherab(self, X, Y, Z) -> np.ndarray:
        """
        Wrapper function for self.calc_emissivity to take inputs from Cherab
        """

        R, phi = XY_To_RPhi(X, Y)
        # --- Convert phi to be positive
        if phi is not None and phi < 0:
            phi += 2.0 * np.pi
        emission = self.calc_emissivity(
            np.array([R]),
            np.array([Z]),
            np.array([phi]),
            emissionName=self.emissionName,
        )
        return emission[self.emissionName].item()

    def _update_bolometer_properties(self) -> None:
        """
        Changes Cherab observation parameters. These should be defined within the radDist
        config file under BOLOMETER_PROPS

        Required values in config file:
        pixelSamples    :: Resolution of the sightline. Higher = better, but takes longer.
        numProcessors   :: The number of processors to use while observing
        """

        boloCameras = self.tokamak.bolometers
        pixelSamples = self.info["BOLOMETER_PROPS"]["pixelSamples"]
        numProcessors = self.info["BOLOMETER_PROPS"]["numProcessors"]

        for bolo_ in boloCameras:
            # --- Either does the top or bottom loop depending on on if there is an extra bolometerCamera layer
            if hasattr(bolo_, "bolometer_camera"):
                foils = list(bolo_.bolometer_camera.foil_detectors)
            elif hasattr(bolo_, "foil_detectors"):
                foils = list(bolo_.foil_detectors)
            else:
                print("---  ERROR. ---")
                print(
                    f"Could not update bolometer properties in RadDist._update_bolometer_properties()"
                )
                print("---  ERROR. ---")
                foils = []
            for foil in foils:
                foil.render_engine.processes = numProcessors
                foil.pixel_samples = pixelSamples

    def _get_scale_factor(self) -> None:
        """
        Creates a dict containing a nested list of the scaling factors for each synthetic signal,
        to be used in the fitting. The list has the same form as the radDist.

        It will first see to see if there is a specific radDist called _scaling_factor, otherwise
        it will returns 1's.

        Form:
        [
            [                               emissionName1
                [bolo1_1, bolo1_2, ...]
                [bolo2_1, bolo2_2, ...],
                ...
            ],
            [                               emissionName2
                [bolo1_1, bolo1_2, ...]
                [bolo2_1, bolo2_2, ...],
                ...
            ],
            ...
        ]
        """

        boloCameras = self.tokamak.bolometers
        scaleFactor = {}
        for emissionName in self.info["emissionNames"]:
            temp = {}
            for bolo_ in boloCameras:
                temp[bolo_.name] = self._scaling_factor(
                    bolo_.info, emissionName=emissionName
                )
            scaleFactor[emissionName] = temp
        self.data["scaleFactor"] = scaleFactor

    @abstractmethod
    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Abstract method to be implemented by subclasses to return the emissivity at the given R, z, and phi.
        """
        pass

    def calc_emissivity(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Broadcast R, z, phi to a common length, then return emissivity from _evaluate.

        Any points outside the tokamak wall are masked to zero.

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates. Each may be length 1
                    (broadcast to the length of the others) or all the same length.
        emissionName : str, optional — defaults to self.emissionName if not given.

        Returns
        -------
        dict of {emissionName: np.ndarray}
        """

        """
        # --- Filling up the other arrays if they are only of length 1
        R = np.asarray(R)
        z = np.asarray(z)
        phi = np.asarray(phi)

        # ── Reconcile R and z lengths ────────────────────────────────────────────
        if len(R) == 1 and len(z) == 1:
            pass  # both scalar, nothing to broadcast
        elif len(R) == 1:
            R = np.full(len(z), R[0])
        elif len(z) == 1:
            z = np.full(len(R), z[0])
        elif len(R) != len(z):
            raise ValueError(
                f"R (len={len(R)}) and z (len={len(z)}) must be the same length, "
                "or one of them must be length 1."
            )

        # ── Broadcast phi to the reconciled length ───────────────────────────────
        n_points = len(R)
        if len(phi) == 1:
            phi = np.full(n_points, phi[0])
        elif len(phi) != n_points:
            raise ValueError(
                f"phi (len={len(phi)}) must be length 1 or match R/z (len={n_points})."
            )
        """

        # --- First call _evaluate to get the emissivity, then check if it is inside the tokamak, if not return 0
        # emiss format: {emissionName: np.array[self._evalutate(R0,z0,phi0) self._evalutate(R1,z1,phi1, ...]}
        # aka, it is an array of emission at each R, z, and phi location
        emiss = self._evaluate(R, z, phi, emissionName=emissionName)

        """
        points = np.column_stack((R, z))
        inside = self.tokamak._inside_tokamak(points)

        # --- Multiply the emissivity by 1 if inside, 0 if outside
        # result = np.zeros_like(inside, dtype=float)
        # result[inside] = 1.0  # WAS 1.0e9 for some reason....

        # Return the emissivity values for each point, setting to 0 if outside the tokamak
        for emissionName in emiss:
            emiss[emissionName][~inside] = 0.0
            # emiss[emissionName] *= result
        """

        return emiss

    def build(self) -> None:
        """
        Creates radDist, finds the power per bin, observes the world for
        each bolometer, then saves the data.

        The tokamak should be created by the individual radDist subtypes
        (helical, elongated ring, etc.) during startup.
        """

        # self.power_per_bin_calc()
        self.data = {}
        self.bolos_observe()
        self._get_scale_factor()
        self.saveRadDist()

    def power_per_bin_calc(
        self, Errfrac: float = 0.01, Pointsupdate: int = int(1e5)
    ) -> None:
        """
        Replaces the power_per_bin_calc with a vectorized version.
        """

        # --- Initilize items
        self.data = {}
        self.data["emisSumArray"] = {}
        self.data["emisSqArray"] = {}
        self.data["powerPerBin"] = {}

        numBins = self.info["numBins"]
        angleperbin = 2.0 * np.pi / numBins

        # Ensure tokamak and its info are initialized
        if not hasattr(self, "tokamak") or self.tokamak is None:
            raise RuntimeError(
                "Tokamak object is not initialized. Call _build_tokamak() before power_per_bin_calc()."
            )
        if not hasattr(self.tokamak, "info") or self.tokamak.info is None:
            raise RuntimeError(
                "Tokamak.info is not initialized. Ensure _build_tokamak() sets up info correctly."
            )

        volumeperbin = self.tokamak.info["MACHINE"]["volume"] / float(numBins)
        pointsperbin = 0
        reachedprecision = 0

        while reachedprecision == 0:
            pointsperbin += Pointsupdate
            # --- Create all of the random points first
            x_, y_, z_, R_, phifirstbin_ = [], [], [], [], []
            while len(x_) < Pointsupdate:
                if self.tokamak.wall is None:
                    raise RuntimeError(
                        "Tokamak wall is not initialized. Ensure that the wall attribute is set before calling power_per_bin_calc()."
                    )
                # Find random point within the wallcurve
                x, y, z, R, phi = Util_radDist.random_uniform_point_noVolume(
                    self.tokamak.wall["wallcurve"],
                    self.tokamak.wall["minr"],
                    self.tokamak.wall["maxr"],
                    self.tokamak.wall["minz"],
                    self.tokamak.wall["maxz"],
                )
                x_.append(x)
                y_.append(y)
                z_.append(z)
                R_.append(R)

                # --- Program rotates all points to be within the first bin,
                # finds the emissivity, then rotates the same points to each
                # subsequent bin
                if phi is not None and phi < 0:
                    phi += 2.0 * np.pi
                phibin = np.floor(phi / angleperbin)
                phifirstbin_.append(phi - (angleperbin * phibin))

            # --- Evalulate all of the points at once
            R_ = np.array(R_)
            z_ = np.array(z_)

            for numbin in range(0, numBins):
                # print(f"Calculating powerPerBin {numbin + 1} of {numBins}")

                # --- use the initial point for each toroidal bin
                phi = np.array(phifirstbin_) + (angleperbin * numbin)

                # Add the emission to the existing arrays
                for emissionName in self.info["emissionNames"]:

                    emission = self.calc_emissivity(
                        R_, z_, phi, emissionName=emissionName
                    )

                    if emissionName not in self.data["emisSqArray"]:
                        self.data["emisSqArray"][emissionName] = np.zeros(numBins)
                        self.data["emisSumArray"][emissionName] = np.zeros(numBins)
                        self.data["powerPerBin"][emissionName] = np.zeros(numBins)

                    self.data["emisSumArray"][emissionName][numbin] += emission[
                        emissionName
                    ].sum()
                    self.data["emisSqArray"][emissionName][numbin] += (
                        emission[emissionName] ** 2
                    ).sum()

            # --- Check to see if the desired precision is reached
            # I believe they are checking to see if the variance between bins is small? -JLH
            emismeanarray = {}
            emisvararray = {}
            integemisarray = {}
            totintegemis = 0
            integemisvararray = {}
            integemiserrarray = {}
            totintegemiserr = 0

            for key_ in self.data["emisSumArray"]:
                emismeanarray[key_] = self.data["emisSumArray"][key_] / pointsperbin
                emisvararray[key_] = (self.data["emisSqArray"][key_] / pointsperbin) - (
                    emismeanarray[key_] ** 2
                )
                integemisarray[key_] = volumeperbin * emismeanarray[key_]

                totintegemis += np.sum(integemisarray[key_])

                integemisvararray[key_] = (
                    volumeperbin**2 * emisvararray[key_] / pointsperbin
                )
                integemiserrarray[key_] = np.sqrt(integemisvararray[key_])
                totintegemiserr += np.sum(integemiserrarray[key_])
            toterrfrac = totintegemiserr / totintegemis

            if toterrfrac < Errfrac:
                reachedprecision = 1
                for key_ in integemisarray:
                    self.data["powerPerBin"][key_] = integemisarray[key_]
            else:
                if pointsperbin % (Pointsupdate * 100) == 0:
                    print(
                        f"Number of points {pointsperbin}, total std. err fraction so far = {toterrfrac:.2e}"
                    )

    def bolos_observe(self) -> None:
        """
        Observes the radiation function for each bolometer.

        Example found here:
        https://www.cherab.info/demonstrations/bolometry/observing_radiation_function.html#bolometer-observing-radiation
        """
        # Should be Power or Radiance
        units = self.info["units"]

        if units == "Power":
            self.data["units"] = "Power [W]"
        elif units == "Radiance":
            self.data["units"] = "Radiance [W / (m2 sr)]"

        # --- Initilize the data storage arrays
        boloCameras = self.tokamak.bolometers
        if not hasattr(self, "data"):
            self.data = {}

        self.data[units] = {}
        self.data[f"{units}_error"] = {}
        self.data[units]["channelOrder"] = {}
        for emissionName in self.info["emissionNames"]:
            self.data[units][emissionName] = {}
            self.data[f"{units}_error"][emissionName] = {}

            for bolo_ in boloCameras:
                self.data[units][emissionName][bolo_.name] = []
                self.data[f"{units}_error"][emissionName][bolo_.name] = []

        # --- Assign sightline resolution, number of processors to be used
        self._update_bolometer_properties()

        # --- Calculate etendue's if asking for radiance
        if units == "Radiance":
            self.tokamak.calc_etendues()

        # --- Populate world with emitter, this cannot be a seperate definition!
        # unless you include the emitter.material changes in that def as well!
        if self.tokamak.wall is None:
            raise RuntimeError(
                "Tokamak wall is not initialized. Ensure that the wall attribute is set before calling bolos_observe()."
            )

        emitter = None
        # --- Add the emitter to the Emission Surface
        for val in self.tokamak.world.children:
            if val.name == "Emission Surface":
                emitter = val

        if emitter is not None:
            emittting_material = VolumeTransform(
                RadiationFunction(self._evalulateCherab), emitter.transform.inverse()
            )
            emitter.material = emittting_material
        else:
            raise RuntimeError(
                "Could not find Tokamak Wall in tokamak.world.children during bolos_observe()."
            )

        # --- Loop over each emission function within the radDist, then each bolometer
        for emissionName in self.info["emissionNames"]:

            # --- Add this to the data arrays
            self.emissionName = emissionName

            # --- Observe with each bolometer
            for bolo_ in boloCameras:
                # print(f"Observing with {bolo_.name}")
                observeVal = []
                observeVal_error = []
                ch_order = []

                # --- Set the units in the foil prior to observing the world
                for jj, foil in enumerate(bolo_.bolometer_camera):
                    ans = 0
                    ans_error = 1.0e3
                    try:
                        foil.units = units
                        foil.observe()
                        if units == "Radiance":
                            # sightline = foil.as_sightline()
                            # sightline.observe()
                            # ans = sightline.pipelines[0].value.mean

                            # --- Below is a calculation of the incident radiance directly,
                            # renormalising for comparison with the sightline
                            try:
                                # --- Check for divide by zero values
                                if (
                                    foil.sensitivity == 0
                                    or foil.pipelines[0].value.mean == 0
                                    or bolo_.etendues[jj] == 0
                                ):
                                    ans = 0
                                    ans_error = 0
                                else:
                                    fractional_solid_angle = (
                                        bolo_.etendues[jj] / foil.sensitivity
                                    )
                                    ans = (
                                        foil.pipelines[0].value.mean
                                        / fractional_solid_angle
                                    )
                                    ans_error = (
                                        np.hypot(
                                            foil.pipelines[0].value.error()
                                            / foil.pipelines[0].value.mean,
                                            bolo_.etendues_error[jj]
                                            / bolo_.etendues[jj],
                                        )
                                        * ans
                                    )
                            except Exception as e:
                                print(
                                    f"An error occured while observing the radiance, {e}"
                                )

                                print(f"Etendue {bolo_.etendues[jj]}")
                                print(f"Sensitivity {foil.sensitivity}")
                                print(
                                    f"foil.pipelines[0].value.mean {foil.pipelines[0].value.mean}"
                                )
                                print(
                                    f"foil.pipelines[0].value.error() {foil.pipelines[0].value.error()}"
                                )

                        elif units == "Power":

                            ans = foil.pipelines[0].value.mean
                            ans_error = foil.pipelines[0].value.error()

                    except Exception as e:
                        print(f"An error occured in bolos_observe: {e}")
                        # print(
                        #    f"Single layer cameras currently not supported, add functionality within bolos_observe!"
                        # )
                    observeVal.append(ans)
                    observeVal_error.append(ans_error)
                    ch_order.append(foil.name)

                # --- Store the data
                self.data[units][emissionName][bolo_.name] = observeVal
                self.data[f"{units}_error"][emissionName][bolo_.name] = observeVal_error
                if bolo_.name not in self.data[units]["channelOrder"]:
                    self.data[units]["channelOrder"][bolo_.name] = ch_order

    def plotCrossSection(self, phi: float = 0.0, ax=None) -> None:
        """
        Returns a contour plot of the radDist at a given phi location
        """

        if self.tokamak.wall is None:
            raise RuntimeError(
                "Tokamak wall is not initialized. Ensure that the wall attribute is set before calling plotCrossSection()."
            )

        RZarray = Util_radDist.callRZGridTokamak(
            self.tokamak, numRgrid=500, numZgrid=500
        )
        R = np.ascontiguousarray(RZarray[:, 0].ravel(), dtype=np.float64)
        z = np.ascontiguousarray(RZarray[:, 1].ravel(), dtype=np.float64)

        emiss = np.zeros(len(RZarray))

        for emissionName in self.info["emissionNames"]:
            ans = self.calc_emissivity(
                R, z, phi=np.array([phi]), emissionName=emissionName
            )
            emiss += ans[emissionName]

        # Create a new figure and axes if ax is None
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        R_unique = RZarray[:, 0]
        z_unique = RZarray[:, 1]
        n_levels = 50

        cf = ax.tricontourf(R_unique, z_unique, emiss, levels=n_levels, cmap="CMRmap_r")
        ax.tricontour(
            R_unique,
            z_unique,
            emiss,
            levels=n_levels,
            colors="white",
            linewidths=0.4,
            alpha=0.4,
        )

        """
        for ii, emissionName in enumerate(self.info["emissionNames"]):
            ax.text(
                rarray[loc_[ii][0]],
                zarray[loc_[ii][1]],
                emissionName,
                zorder=1,
                ha="center",
                va="center",
                color="black",
                weight="bold",
            )
        """

    def plotOverview(self, return_figure: bool = False, plot_etendue: list = []):
        """
        Plots the bolometer chords with a contour overplot of the radDist in the top row
        Plots the observed emissivities in the bottom row

        return_figure :: True = returns the figure object

        """

        tok = self.tokamak

        # --- Make the emission surface transparent for accurate ray tracing
        tok._make_raysect_surface_transparent(surfaceName="Emission Surface")

        # --- Plot everything ---
        # Top row: every bolometer with radDist contour overplot + injection location radDist on the right

        colors = ["black", "purple", "blue", "green", "orange", "red"]
        if tok.info is not None:
            boloGroups = tok.info["Bolometer Groups"]
            bolometers = tok.bolometers

            num_columns = len(boloGroups) + 1
            num_rows = 2
            f = plt.figure(figsize=(15, 8))
            plot_count = 0

            # --- Loop over each bolometer group
            for ii, boloGroup in enumerate(boloGroups):
                plot_count += 1
                f_ = f.add_subplot(num_rows, num_columns, plot_count)
                tok._plot_first_wall(f_)
                tok._plot_bolometers(
                    f_,
                    boloGroupName=boloGroup,
                    plot_chord_info=False,
                    plot_etendue=plot_etendue,
                    legend=False,
                )

                # --- Plot a cross-section of the radDist
                phi = tok.get_ave_bolometer_tor_loc(boloGroupName=boloGroup)
                if phi is not None:
                    self.plotCrossSection(phi=np.deg2rad(phi), ax=f_)

            # --- Plot the injection location
            plot_count += 1
            phi = self.info["injectionLocation"]
            f_ = f.add_subplot(num_rows, num_columns, plot_count)
            tok._plot_first_wall(f_)
            self.plotCrossSection(phi=np.deg2rad(phi), ax=f_)
            f_.set_title(f"Injection Location, phi = {phi} degrees")

            # --- Plot the observed emissivities
            for ii, boloGroup in enumerate(boloGroups):
                plot_count += 1
                f_ = f.add_subplot(num_rows, num_columns, plot_count)

                for qq, emissionName in enumerate(self.info["emissionNames"]):
                    # --- Group the data
                    data_ = []
                    chan_ = []

                    for jj, bolo in enumerate(bolometers):
                        if bolo.info["GROUP_NAME"] == boloGroup:
                            ch_tags = bolo.info["CHANNEL_TAGS"]
                            c_ = []
                            for ch in ch_tags:  # type: ignore
                                c_.append(int(ch[-2:]))

                            data_ += self.data[self.info["units"]][emissionName][
                                bolo.info["NAME"]
                            ]
                            chan_ += c_

                    # --- Sort the channel list in ascending order
                    inds = np.array(chan_).argsort()
                    f_.plot(
                        np.array(chan_)[inds],
                        np.array(data_)[inds],
                        color=colors[qq],
                        label=emissionName,
                    )

                f_.legend()
                f_.set_ylim(0, f_.get_ylim()[1])
                f_.set_ylabel(f"{self.data['units']}")
                f_.set_xlabel("channel")
                f_.set_title(boloGroup)

            plt.tight_layout()

            if return_figure:
                return f
            else:
                plt.show()

    def saveRadDist(self) -> None:
        """
        Saves the data and radDist information
        """

        # --- Convert items within the self.info and self.data to lists
        toSave = {
            "info": convert_arrays_to_list(self.info),
            "data": convert_arrays_to_list(self.data),
        }

        # --- Save the data
        folderName = f"{self.info['distType']}_elongation_{self.info['elongation']}_polSigma_{self.info['polSigma']}_rotation{self.info['rotationAngle']}"
        saveFileName = f"R_{self.info['startR']:.2f}_z_{self.info['startZ']:.2f}.json"

        pathFileName = os.path.join(
            EMIS3D_INPUTS_DIRECTORY,
            self.info["tokamakName"],
            "radDists",
            self.info["saveRunsDirectoryName"],
            folderName,
        )

        save_json(toSave, pathFileName, saveFileName)

    @abstractmethod
    def _scaling_factor(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
        """
        Abstract method to be implemented by subclasses to return the
        scaling factor for the bolometer.
        """


class Helical(RadDist):
    """
    Helical radDist class used to produce radDist based on the magnetic
    field line at a given R and z.

    Parameters
    ----------
    startR, startz  : float     — poloidal start location in meters.
    config          : dict      — configuration dictionary.
    setFieldLine    : bool      — Wether to trace the field lines on construction.
                                  Should be True in almost all cases
    """

    def __init__(
        self,
        startR: float = 0.0,
        startZ: float = 0.0,
        config: dict = {},
        setFieldLine: bool = True,
    ) -> None:

        super(Helical, self).__init__(startR=startR, startZ=startZ, config=config or {})

        self.info["setFieldLine"] = setFieldLine
        self.info["distType"] = "helical"

        # --- Create the field line to trace
        if setFieldLine:

            print(
                f"Building Helical radDist | "
                f"polSigma={self.info['polSigma']:.2f}, "
                f"R={startR:.2f} m, z={startZ:.2f} m"
            )

            self._build_tokamak(
                tokamakName=self.info["tokamakName"],
                mode="Build",
                reflections=False,
                eqFileName=self.info["eqFileName"],
            )
            self.setFieldLine()

    def setFieldLine(self) -> None:
        """
        Sample source points from the bivariate normal convolution and
        trace the corresponding field lines through the tokamak.
        """

        R0 = self.info["startR"]
        z0 = self.info["startZ"]
        start_phi = self.info["startPhiRad"]
        sigma_kernel = self.info["sigmaKernel"]
        num_lines = self.info["numFieldLines"]

        # sigma_target must come from config; fall back to a small offset only
        # if not provided so the intent is explicit rather than a magic number.
        sigma_target = self.info.get("sigmaTarget", sigma_kernel + 0.01)
        self.info["sigma_target"] = sigma_target

        points, weights = Util_radDist.bivariate_normal_isodensity_points(
            R0, z0, sigma_target, sigma_kernel, num_lines, seed=42
        )

        self.info["weights"] = weights

        self.tokamak.set_fieldlines(
            startR=points[:, 0],
            startZ=points[:, 1],
            startPhi=start_phi,
            numTransists=1.0,
        )

        startPhideg = str(int(np.rad2deg(start_phi)))

        if startPhideg not in self.tokamak.fieldLines:
            raise RuntimeError(
                f"startPhiRad={start_phi:.4f} rad ({startPhideg}°) not found in "
                f"traced field lines. Available: "
                f"{self.tokamak.get_fieldLines_startPhis()}"
            )

        self.info["emissionNames"] = self.tokamak.fieldLines[startPhideg][
            "directionNames"
        ]
        self.info["numTransists"] = 1.0

    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the point (R,z,ph).

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates.
        emissionName : str, optional — defaults to self.emissionName if not given.
        """

        # --- Set emissionName, this happens when self._evalulateCherab() is used
        if emissionName is None:
            emissionName = self.emissionName

        R0_arr, z0_arr = self.tokamak.find_RZ_Fline(
            str(self.info["startPhi"]), emissionName, inputPhis=phi
        )

        # Squeeze out any spurious dimensions, ensure C-contiguous float64
        R = np.ascontiguousarray(R.ravel(), dtype=np.float64)
        z = np.ascontiguousarray(z.ravel(), dtype=np.float64)
        R0_arr = np.ascontiguousarray(R0_arr.ravel(), dtype=np.float64)
        z0_arr = np.ascontiguousarray(z0_arr.ravel(), dtype=np.float64)

        tot = Util_radDist._evaluate_kernels(
            R,
            z,
            R0_arr,
            z0_arr,
            self.info["weights"].astype(np.float64),
            float(self.info["polSigma"]),
        )

        return {emissionName: tot}

    def _scaling_factor(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
        """
        Compute the toroidal scaling factor (phi offset) for a bolometer channel.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        num_channels = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        rev_number = 0
        if emissionName is not None and "rev" in emissionName:
            rev_number = int(emissionName.split("rev")[-1])

        return [phi + rev_number * 2.0 * np.pi] * num_channels


class HelicalRing(RadDist):
    """
    HelicalRing radDist class used to produce radDist based on the magnetic
    field line at a given R and z.

    Parameters
    ----------
    startR, startz  : float     — centre of the field line.
    config          : dict      — dict of the configuration file.
    setFieldLine    : bool      — Trace the field line, should be true 99% of the time.
    """

    def __init__(
        self,
        startR: float = 0.0,
        startZ: float = 0.0,
        config: dict = {},
        setFieldLine: bool = True,
    ):

        super(HelicalRing, self).__init__(
            startR=startR, startZ=startZ, config=config or {}
        )

        self.info["setFieldLine"] = setFieldLine
        self.info["distType"] = "helical"

        # --- Create the field line to trace
        if setFieldLine:
            str_ = f"Building HelicalRing radDist using a polSigma of {self.info['polSigma']:.2f} elongation of {self.info['elongation']:.2f},"
            str_ += f" starting at R = {startR:.2f}m and z = {startZ:.2f}m"
            print(str_)
            self._build_tokamak(
                tokamakName=self.info["tokamakName"],
                mode="Build",
                reflections=False,
                eqFileName=self.info["eqFileName"],
            )
            self.setFieldLine()

    def setFieldLine(self) -> None:
        """
        Traces the field line based on the startR and startZ
        """
        numTransists = 1.0

        self.tokamak.set_fieldlines(
            startR=[self.info["startR"]],
            startZ=[self.info["startZ"]],
            startPhi=self.info["startPhiRad"],
            numTransists=numTransists,
        )
        startPhideg = f'{int(np.rad2deg(self.info["startPhiRad"]))}'

        if startPhideg not in self.tokamak.fieldLines:
            raise RuntimeError(
                f"Input fieldLinePhi of {startPhideg}, not availble!"
                f"Possible fieldLinePhi(s): {self.tokamak.get_fieldLines_startPhis()}"
            )
        self.info["emissionNames"] = self.tokamak.fieldLines[startPhideg][
            "directionNames"
        ]
        self.info["numTransists"] = numTransists

    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the point (R,z,ph).

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates.
        emissionName : str, optional — defaults to self.emissionName if not given.
        """

        # --- Set emissionName if called with self._evalulateCherab()
        if emissionName is None:
            emissionName = self.emissionName

        localEmis = {}
        R0, z0 = self.tokamak.find_RZ_Fline(
            str(self.info["startPhi"]), emissionName, inputPhis=phi
        )
        R0 = R0.flatten()
        z0 = z0.flatten()

        vertExtendParam = 3.0  # for vertical extension of plasma... hardcoded for now

        # next we need the R,Z position of our helical structure at this phi
        flR, flZ = self.tokamak.find_RZ_Fline(
            str(self.info["startPhi"]), emissionName, inputPhis=phi
        )

        # now for bivariate normal distribution in poloidal plane.
        # elongated in approximate poloidal direction of field line

        # first we need to decompose (R,Z) in terms of parallel/perpendicular
        # to approximate field line. Approximated as the perpendicular direction
        # to the vector from (major radius, zoffset) to (flR, flZ)
        # "cent0" = (major radius, zoffset), "cent1" = (flR, flZ), "point" = (R,Z)
        if self.tokamak.info is not None:
            cent0ToCent1Vec = [flR - self.tokamak.info["MACHINE"]["majorRadius"], flZ]
            cent0ToCent1Vec[1] = cent0ToCent1Vec[1] / vertExtendParam
            cent0ToCent1VecMag = np.sqrt(
                cent0ToCent1Vec[0] ** 2 + cent0ToCent1Vec[1] ** 2
            )
            cent0ToCent1VecNormed = [x / cent0ToCent1VecMag for x in cent0ToCent1Vec]
            perpVecNormed = [-cent0ToCent1VecNormed[1], cent0ToCent1VecNormed[0]]
            cent1ToPointVec = [R - flR, z - flZ]
            paralleldist = (
                cent1ToPointVec[0] * cent0ToCent1VecNormed[0]
                + cent1ToPointVec[1] * cent0ToCent1VecNormed[1]
            )
            perpdist = (
                cent1ToPointVec[0] * perpVecNormed[0]
                + cent1ToPointVec[1] * perpVecNormed[1]
            )

            emis = (
                (
                    1.0
                    / (
                        2.0
                        * np.pi
                        * self.info["elongation"]
                        * (self.info["polSigma"] ** 2)
                    )
                )
                * np.exp(
                    -0.5
                    * (perpdist**2)
                    / (self.info["polSigma"] * self.info["elongation"]) ** 2
                )
                * np.exp(-0.5 * (paralleldist**2) / self.info["polSigma"] ** 2)
            )

            localEmis[emissionName] = emis.flatten()

        return localEmis

    def _scaling_factor(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
        """
        Compute the toroidal scaling factor (phi offset) for a bolometer channel.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        num_channels = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        rev_number = 0
        if emissionName is not None and "rev" in emissionName:
            rev_number = int(emissionName.split("rev")[-1])

        return [phi + rev_number * 2.0 * np.pi] * num_channels


class ElongatedRing(RadDist):
    """
    Elongated Ring radDist class used to produce radDist based on the input
    R, z, polSigma, and elongation.

    INPUTS:

    numBins :: The number of toroidal bins
    """

    def __init__(
        self,
        startR=None,
        startZ=None,
        config={},
    ):
        # Ensure startR and startZ are floats, not None
        if startR is None:
            startR = float(config.get("startR"))
        if startZ is None:
            startZ = float(config.get("startZ"))

        super(ElongatedRing, self).__init__(
            startR=startR, startZ=startZ, config=config or {}
        )
        self.info["distType"] = "elongatedRing"
        self.info["emissionNames"] = ["elongatedRing"]

        if "polSigma" in self.info:
            self._build_tokamak(
                tokamakName=self.info["tokamakName"],
                mode="Build",
                reflections=False,
                eqFileName=self.info["eqFileName"],
            )

            str_ = f"Building Elongated Ring radDist using a polSigma of {self.info['polSigma']:.2f}"
            str_ += f", elongation of {self.info['elongation']:.2f}, rotation angle of {self.info['rotationAngle']:.2f}, starting at R = {startR:.2f}m and z = {startZ:.2f}"

            print(str_)

    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the point (R,z,ph).

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates.
        emissionName : str, optional — defaults to self.emissionName if not given.
        """
        # --- Set emissionName if called with self._evalulateCherab()
        if emissionName is None:
            emissionName = self.emissionName

        localEmis = {}
        # bivariate normal distribution in poloidal plane.
        # integrated over dR and dZ this function returns 1. I think.
        localEmis[emissionName] = Util_radDist.bivariate_normal_elongated(
            R=R,
            R0=self.info["startR"],
            z=z,
            z0=self.info["startZ"],
            elongation=self.info["elongation"],
            pol_sigma=self.info["polSigma"],
        )
        return localEmis

    def _scaling_factor(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
        """
        Compute the toroidal scaling factor (phi offset) for a bolometer channel.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        numChan = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [float(phi)] * numChan


class SquareTube(RadDist):
    """
    Produces a square tube around the torus. This class is designed to test the
    foil.observe() functionality with a simple geometry and make sure that things are correct.

    """

    def __init__(
        self,
        startR=None,
        startZ=None,
        config={},
    ):
        # Ensure startR and startZ are floats, not None
        if startR is None:
            startR = float(config.get("startR"))
        if startZ is None:
            startZ = float(config.get("startZ"))

        dR = config.get("dR")
        dz = config.get("dz")
        if dR is None or dz is None:
            raise ValueError(
                "dR and dz must be provided in the config for SquareTube radDist."
            )

        super(SquareTube, self).__init__(
            startR=startR, startZ=startZ, config=config or {}
        )
        self.info["distType"] = "squareTube"
        self.info["emissionNames"] = ["squareTube"]

        self._build_tokamak(
            tokamakName=self.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=self.info["eqFileName"],
        )

        str_ = f"Building Square Tube radDist using starting at R = {startR:.2f} +/- {dR:.2f}m and z = {startZ:.2f} +/- {dz:.2f}m"
        print(str_)

    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the point (R,z,ph).

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates.
        emissionName : str, optional — defaults to self.emissionName if not given.
        """

        # --- Set emissionName if called with self._evalulateCherab()
        if emissionName is None:
            emissionName = self.emissionName

        localEmis = {}
        # --- Return 1 if within the square tube, 0 if outside
        R_in = (R >= self.info["startR"] - self.info["dR"]) & (
            R <= self.info["startR"] + self.info["dR"]
        )
        z_in = (z >= self.info["startZ"] - self.info["dz"]) & (
            z <= self.info["startZ"] + self.info["dz"]
        )
        loc_ = np.where(R_in & z_in)
        emission = np.zeros(len(R))
        emission[loc_] = 1.0

        localEmis[emissionName] = emission

        return localEmis

    def _scaling_factor(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
        """
        Compute the toroidal scaling factor (phi offset) for a bolometer channel.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        numChan = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [float(phi)] * numChan

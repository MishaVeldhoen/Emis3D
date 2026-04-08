# -*- coding: utf-8 -*-
"""
Main solving routine. The radDist data will be loaded and organized
based on how the program observed each bolometer:

------------------------------------------------------------------------------------------------
synthetic = [
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ],   # injection location 1
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ],   # injection location 2
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ],   # injection location 3,
        ...
]

data = [
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ] # nested list of each bolometer array
]

data_err = [
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ] # nested list of each bolometer array
]

scale = [
           [s1, s2, ...], # Scaling factor for each bolometer based on injection location 1
           [s1, s2, ...], # Scaling factor for each bolometer based on injection location 2
]

This scale array is purely optional and will be set to 1 if not specified. It is used for
some scaling function definitions (such as distance from the injection location for the
helical radDist).
------------------------------------------------------------------------------------------------



The overall minimization function is organized as such:
------------------------------------------------------------------------------------------------
res = ((data - scale_function(params, scale) * synthetic) / error)

If you have multiple injection locations:
res = ((data - scale_function(params_1, scale_1) * synthetic_1) / error) +
      ((data - scale_function(params_2, scale_2) * synthetic_2) / error) + ...

The LMFIT minimzation routine takes care of squaring the residual, that is why we don't do it here
------------------------------------------------------------------------------------------------

Re-organised during the refactor - JLH Aug., 2025


REMINDERS:
1. Make sure that the pre-processed SXR/Bolometer data are in the same units as those when the radDists were created
2. The bolometers have different responsivities with respect to each other! Some pre-analysis of the bolometer data is necessary
in order to scale them relative to each other. Then you should load the processed data in this program.
3. This program uses a right-handed coordinate system (positive phi in counter-clockwise direction when looking down).
So you need to offset your angles by 360 - x from DIII-D coordinates.
4. crossCalib Flag set to True should only be used to find cross-calibration factors between bolometers for a specific shot.
This is typically done during the current quench, when the radiation is assumed to be axisymmetric. The flag allows for each
bolometer to have indpendent scaling factors, see the example on how to cross-calibrate uncalibrated bolometers.


TODO:
1. Prepare fits -> Write definition to combine multiple locationDependent values together,
see _combine_synthetics_for_fits as a older starting point
2. Give the user the option to use the new error technique or use the error from the data
3. Double check field line tracer with output from MOFAT
4. See if crossCalib: is needed in the _preform_fits definition

Biggest Issues:
1. Find good radDist functions that represent what is going on
    - Have the helical distribution change orientation (and shape?) as it goes around the vessel
    - Add a tomography radDist mapping function (like BOLT?)
2. Implement a toroidal distribution function that is not symmetric around the injection loction
3. Re-vist how error is calculated for the observed data


BUG:
1. Program crashes when using an averaging window of 0.0001 for JET data? Is this smaller than the time resolution
or something?

"""

import os
import time

import dill
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, report_fit
from scipy.integrate import simpson

import main.Util_emis3D as Util_emis3D
from main.Globals import *
from main.Tokamak import Tokamak
from main.Util import (
    config_loader,
    convert_arrays_to_list,
    find_max_nested_lists,
    get_filenames_in_directory,
    read_h5,
)
from main.radDistFitting import RadDistFitting
from main.radDist import Helical, ElongatedRing, HelicalRing


class Emis3D:

    def __init__(
        self,
        tokamakName: str | None = None,
        runConfigName: str | None = None,
        initialize: bool = True,
        verbose: bool = False,
    ):
        """
        Main class for running the emis3D program.

        Parameters
        ----------
        tokamakName   : Name of the tokamak
        runConfigName : Name of the run configuration file
        initialize    : When True the standard initialisation pipeline is run
        verbose       : Print extra progress information
        """

        # --- Initialize variables
        self.data = {}
        self.info = None
        self.verbose = verbose
        self.error_free = True

        if initialize:
            self._initialize(tokamakName=tokamakName, runConfigName=runConfigName)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(
        self, tokamakName: str | None = None, runConfigName: str | None = None
    ) -> None:
        """Load configuration, bolometer data and radDists."""
        if not self.error_free:
            print("An error occurred, cannot run the program")
            return

        if tokamakName is None or runConfigName is None:
            print("tokamakName or runConfigName is None, cannot run the program")
            return

        self._load_config_file(tokamakName=tokamakName, runConfigName=runConfigName)
        self._load_bolometer_data()
        self._create_master_channel_order()
        self._load_radDists()

    def _perform_fits(self, evalTime: float, crossCalib: bool = False) -> None:
        """
        Prepares the data, preforms the fits

        Parameters
        ----------
        evalTime   : Time to run the fit
        crossCalib : Flag to tell the program to find the calibration
                     factor between bolometers during the CQ
        """
        if self.error_free:
            self._prepare_fits(evalTime=evalTime, crossCalib=crossCalib)
            t_start = time.time()

            """
            NOTE: IS THIS NEEDED ANYMORE?
            if crossCalib:
                self._minimize_radDists(evalTime=evalTime)
            else:
                if (
                    self.info is not None
                    and self.info.get("numProcessorsFitting", 1) > 1
                ):
                    self._run_parallel(
                        evalTime=evalTime, max_workers=self.info["numProcessorsFitting"]
                    )
                else:
            """

            self._minimize_radDists(evalTime=evalTime)
            print(f"Fitting done in {time.time() - t_start:.2f} seconds")
            self._post_process_fit_arrangement(evalTime=evalTime)
            self._post_process_radiation_distribution(evalTime=evalTime)
            self._post_process_calculations(evalTime=evalTime)

    # ------------------------------------------------------------------
    # Configuration / data loading
    # ------------------------------------------------------------------

    def _load_config_file(
        self,
        tokamakName: str | None = None,
        runConfigName: str | None = None,
        pathFileName: str | None = None,
    ) -> None:
        """
        Loads the YAML configuration file for the given tokamak

        Either supply ``pathFileName`` directly, or supply both
        ``tokamakName`` and ``runConfigName`` to have the path constructed
        automatically.
        """

        # --- Only run if error free
        if not self.error_free:
            return

        try:
            if pathFileName is None:
                if tokamakName is not None and runConfigName is not None:
                    pathFileName = join(
                        EMIS3D_INPUTS_DIRECTORY, tokamakName, "runs", runConfigName
                    )
                else:
                    raise Exception(
                        "tokamakName, runConfigName, or pathFileName is None; "
                        "cannot load config file"
                    )

            # --- Load the configuration file, if it exists
            if not os.path.isfile(pathFileName):
                raise Exception(f"File does not exist: {pathFileName}")

            self.info = config_loader(pathFileName, verbose=self.verbose)

            # --- Raise exception if the file failed to load
            if self.info is None:
                raise Exception(
                    f"Could not load the configuration file: {pathFileName}"
                )

            # --- Store the tokamak and runConfig name
            self.info["tokamakName"] = tokamakName
            self.info["runConfigName"] = runConfigName

        except Exception as e:
            print(f"An error occurred loading the config file:\n{e}")
            self.error_free = False

    def _load_radDists(self) -> None:
        """
        Loads each radDist within the directories listed in the config file:
        radDistDirectories_LocIndependent, and radDistDirectories_LocDependent
        """

        # --- Only continue if error free
        if not self.error_free:
            return

        if self.verbose:
            print("Loading radDists")

        try:
            # --- Initilize synthetic signal arrays
            self.data.update({"synthetic": {"locDependent": {}, "locIndependent": {}}})

            # --- Find the files, store as a nested list
            self.files = []

            if self.info is None or (
                "radDistDirectories_LocIndependent" not in self.info
                and "radDistDirectories_LocDependent" not in self.info
            ):
                print("No radDistDirectories found in the config file")
                return

            # --- Load location independent radDists
            dirs_ = self.info["radDistDirectories_LocIndependent"]
            count_ = 0

            for dir_ in dirs_:
                pathFileName = os.path.join(
                    EMIS3D_INPUTS_DIRECTORY,
                    self.info["tokamakName"],
                    "radDists",
                    dir_,
                )

                # --- Loop over the files and load the radDist
                for file_ in get_filenames_in_directory(pathFileName):
                    try:

                        temp_ = RadDistFitting(radDistPath=file_)
                        self.data["synthetic"]["locIndependent"][count_] = temp_
                        count_ += 1
                    except Exception as e:
                        print(f"Error loading radDist {file_}: {e}")

            # --- Load radDists that are location independent
            dirs_ = self.info["radDistDirectories_LocDependent"]

            for ii, dir_ in enumerate(dirs_):
                self.data["synthetic"]["locDependent"][f"loc_{ii}"] = {}

                pathFileName = os.path.join(
                    EMIS3D_INPUTS_DIRECTORY,
                    self.info["tokamakName"],
                    "radDists",
                    dir_,
                )

                # --- Loop over the files and load the radDist
                count_ = 0
                for file_ in get_filenames_in_directory(pathFileName):
                    try:
                        temp_ = RadDistFitting(radDistPath=file_)
                        self.data["synthetic"]["locDependent"][f"loc_{ii}"][
                            count_
                        ] = temp_
                        count_ += 1
                    except Exception as e:
                        print(f"Error loading radDist {file_}: {e}")

            if self.verbose:
                print("Done loading radDists")

        except Exception as e:
            print(f"An error occured while loading synthetic data: {e}")
            self.error_free = False

    def _load_bolometer_data(self) -> None:
        """
        Load pre-calibrated SXR/bolometer data from ``inputs/{tokamakName}/sxrData/``.
        The filename is taken from ``dataFileName`` in the run config.
        """

        # --- Only run if error free
        if not self.error_free:
            return

        try:
            # --- Exit if there is no config file loaded
            if self.info is None or "BOLOMETERS" not in self.info:
                raise Exception("No BOLOMETERS found in the config file")

            # --- Load the data
            self.data["observed"] = {}
            for bolo_ in self.info["BOLOMETERS"]:
                pathFileName = os.path.join(
                    EMIS3D_INPUTS_DIRECTORY,
                    self.info["tokamakName"],
                    "sxrData",
                    self.info["BOLOMETERS"][bolo_]["dataFileName"],
                )
                if not os.path.isfile(pathFileName):
                    raise FileNotFoundError(f"File does not exist: {pathFileName}")

                if self.verbose:
                    print(f"Loading bolometer data: {pathFileName}")

                temp_ = read_h5(pathFileName)

                # --- Decode channel names from bytes
                for ii, ch_ in enumerate(temp_["channelOrder"]):
                    temp_["channelOrder"][ii] = ch_.decode("utf-8")

                # --- Apply optional scaling factor
                temp_["DATA_CALIBRATED"] *= self.info["BOLOMETERS"][bolo_].get(
                    "scalingFactor", 1.0
                )

                # --- Store the data
                self.data["observed"][bolo_] = temp_

        except Exception as e:
            print(f"An error occurred loading the bolometer data:\n{e}")
            self.error_free = False

    def _create_master_channel_order(self) -> None:
        """
        Build a master channel list from the loaded bolometer data.
        All radDists and raw data will be ordered to match this list.
        """

        # --- Only run if error free
        if not self.error_free:
            return

        try:
            if self.info is None or "BOLOMETERS" not in self.info:
                raise Exception("No BOLOMETERS found in the config file")

            # --- Initilize the arrays
            self.channel_order = {
                "bolometer_order": [],
                "channel_list": [],
            }

            for bolo in self.data["observed"]:
                self.channel_order["bolometer_order"].append(bolo)
                self.channel_order["channel_list"].append(
                    list(self.data["observed"][bolo]["channelOrder"])
                )

                """
                # OLD:
                temp_ = []
                for channel in self.data["observed"][bolo]["channelOrder"]:
                    temp_.append(channel)
                self.channel_order["channel_list"].append(temp_)
                """

        except Exception as e:
            print(f"An error occurred creating the master channel order:\n{e}")
            self.error_free = False

    # ------------------------------------------------------------------
    # Fit preparation
    # ------------------------------------------------------------------

    def _average_observed_data(
        self, arrayName: str = "", evalTime: float | None = None
    ) -> list:
        """
        Average SXR/bolometer data over a window [evalTime-dt, evalTime+dt].

        Returns a flat list of averaged channel values, or zeros if evalTime
        is None or dt is not configured.
        """

        if evalTime is None:
            return [0] * self.data["observed"][arrayName]["NUM_CHANNELS"]

        time_ = self.data["observed"][arrayName]["TIME"]

        if self.info is None or "dt" not in self.info:
            return [1] * self.data["observed"][arrayName]["NUM_CHANNELS"]

        dt_ = self.info["dt"]
        start = np.abs(time_ - (evalTime - dt_)).argmin()
        end = np.abs(time_ - (evalTime + dt_)).argmin()

        vals = np.mean(
            self.data["observed"][arrayName]["DATA_CALIBRATED"][:, start:end],
            axis=1,
        )

        # --- Replace NaNs with zero
        vals = np.where(np.isnan(vals), 0.0, vals)

        return convert_arrays_to_list(vals)

    def _prepare_data_for_fit(self, evalTime: float) -> None:
        """
        Average and organise observed data for the minimisation fit.

        Creates
        -------
        self.fitData[evalTime]['observed']       : Averaged calibrated data
        self.fitData[evalTime]['observed_error'] : Error estimates
        """

        # --- Only run if error free
        if not self.error_free:
            return

        try:
            if not hasattr(self, "fitData"):
                self.fitData = {}

            # NOTE: observed is what is used in the minimization process.
            #       bolo data is organzied to make plotting easier

            self.fitData[evalTime] = {
                "observed": [],
                "observed_error": [],
                # --- Store the data for each bolometer,
                "boloData": {},
                "boloData_error": {},
            }

            # --- Average and map the data to a dict
            data_ = {}
            for bolo_ in self.data["observed"]:
                temp = self._average_observed_data(arrayName=bolo_, evalTime=evalTime)
                data_.update(
                    dict(zip(self.data["observed"][bolo_]["channelOrder"], temp))
                )

            self.fitData[evalTime]["dataMap"] = data_

            # --- Find the max value in all the channels
            max_ = max(
                (
                    data_[ch]
                    for channels in self.channel_order["channel_list"]
                    for ch in channels
                ),
                default=0.0,
            )

            for ii, channels in enumerate(self.channel_order["channel_list"]):
                temp = []
                temp_e = []

                bolo_ = self.channel_order["bolometer_order"][ii]

                # --- Find the max value for that specific array
                """
                max_ = 0
                for channel in channels:
                    if data_[channel] > max_:
                        max_ = data_[channel]
                """

                for channel in channels:
                    val = data_[channel]
                    temp.append(val)

                    # --- Large error for zero signal
                    if val > 1.0:
                        err_frac = Util_emis3D.error_exponential(
                            val, max_, scale_factor=1.0
                        )
                        err_ = val * err_frac
                    else:
                        err_ = np.float64(1.0e4)

                    temp_e.append(err_)

                self.fitData[evalTime]["observed"].append(temp)
                self.fitData[evalTime]["observed_error"].append(temp_e)

                # --- Temp is a nested list of bolometers [[Bolo1-1, Bolo1-2, ...], [Bolo2-1, Bolo2-2], etc.]
                self.fitData[evalTime]["boloData"][bolo_] = temp
                self.fitData[evalTime]["boloData_error"][bolo_] = temp_e

            if self.verbose:
                print(f"Observed data prepared for fitting")

        except Exception as e:
            print(f"An error occured while preparing data for the fit: {e}")
            self.error_free = False

    def _prepare_synthetic_for_fits(
        self, evalTime: float, crossCalib: bool = False
    ) -> None:
        """Prepare and parameterise every radDist for the minimisation."""

        # --- Only run if error free
        if not self.error_free:
            return

        try:
            print("Preparing synthetic data for fitting")
            max_data_val = find_max_nested_lists(self.fitData[evalTime]["observed"])
            if self.info is not None and "varyScaleFactor" in self.info:

                boloNames = (
                    self.channel_order["bolometer_order"] if crossCalib else None
                )

                # --- Arrange and create parameters for the location dependent data
                for loc in self.data["synthetic"]["locDependent"]:
                    for number_ in self.data["synthetic"]["locDependent"][loc]:
                        radD = self.data["synthetic"]["locDependent"][loc][number_]
                        radD.prepare_for_fits(
                            self.channel_order["channel_list"],
                            data_max=max_data_val,
                        )

                        radD.create_parameters(
                            boloNames=boloNames,
                            varyScaleFactor=self.info["varyScaleFactor"],
                        )

                # --- Arrange and create parameters for the location independent data
                for number_ in self.data["synthetic"]["locIndependent"]:
                    radD = self.data["synthetic"]["locIndependent"][number_]
                    radD.prepare_for_fits(
                        self.channel_order["channel_list"], data_max=max_data_val
                    )
                    radD.create_parameters(
                        boloNames=boloNames,
                        varyScaleFactor=self.info["varyScaleFactor"],
                    )

            if self.verbose:
                print("Done preparing synthetic data for fit")

        except Exception as e:
            print(f"An error occured while preparing synthetic data for fitting: {e}")
            self.error_free = False

    def _build_synthetic_dict(
        self, radDist_, locDependence: str, number, location=None
    ) -> dict:
        """
        Helper: build the synthetic_dict entry shared by both locIndependent
        and locDependent radDist loops in _prepare_fits.
        """
        if location is None:
            source = self.data["synthetic"][locDependence][number]
        else:
            source = self.data["synthetic"][locDependence][location][number]

        synthetic_dict = {
            "paramName": radDist_.fitSynthetic["params"]["paramName"],
            "injectionLocation": source.info["injectionLocation"],
            "injectionLocation_rad": np.deg2rad(source.info["injectionLocation"]),
            "emissionNames": radDist_.info["emissionNames"],
        }

        for emissionName in radDist_.info["emissionNames"]:
            synthetic_dict[emissionName] = {
                "scaleSynth": radDist_.fitSynthetic[emissionName]["scaleSynth"],
                "scaleFactor": radDist_.fitSynthetic[emissionName]["scaleFactor"],
                "data": radDist_.fitSynthetic[emissionName]["data"],
                "data_error": radDist_.fitSynthetic[emissionName]["data_error"],
            }

        return synthetic_dict

    def _prepare_fits(self, evalTime: float, crossCalib: bool = False) -> None:
        """
        Organise data and synthetic signals into self.fits[evalTime] in
        preparation for _minimize_radDists.
        """
        # --- Only run if error free
        if not self.error_free:
            return

        try:
            if self.verbose:
                print("Preparing data for fitting")

            self._prepare_data_for_fit(evalTime=evalTime)
            self._prepare_synthetic_for_fits(evalTime=evalTime, crossCalib=crossCalib)

            if self.verbose:
                print("Arranging radDists for fitting")

            if not hasattr(self, "fits"):
                self.fits = {}

            # --- Initilze the fitting dictionary
            self.fits[evalTime] = {}
            fitCount = -1

            # --- Location-independent radDists
            for number in self.data["synthetic"]["locIndependent"]:
                fitCount += 1
                radDist_ = self.data["synthetic"]["locIndependent"][number]

                self.fits[evalTime][fitCount] = {
                    "info": {
                        "radDists": {
                            "locationdependence": "locIndependent",
                            "radDistNumber": number,
                            "location": None,
                        }
                    },
                    "parameters": radDist_.fitSynthetic["params"]["params"],
                    "synthetic_dict": self._build_synthetic_dict(
                        radDist_, "locIndependent", number
                    ),
                }

            # --- Location-dependent radDists
            for loc_ in self.data["synthetic"]["locDependent"]:
                for number in self.data["synthetic"]["locDependent"][loc_]:
                    fitCount += 1
                    radDist_ = self.data["synthetic"]["locDependent"][loc_][number]

                    self.fits[evalTime][fitCount] = {
                        "info": {
                            "radDists": {
                                "locationdependence": "locDependent",
                                "radDistNumber": number,
                                "location": loc_,
                            }
                        },
                        "parameters": radDist_.fitSynthetic["params"]["params"],
                        "synthetic_dict": self._build_synthetic_dict(
                            radDist_, "locDependent", number, location=loc_
                        ),
                    }

            num_fits = len(self.fits[evalTime])
            if self.info is not None:
                self.info["numFits"] = num_fits
            self.fits[evalTime]["chiSqVec"] = np.full(num_fits, 1.0e19)

        except Exception as e:
            print(f"An error occured while preparing the fits: {e}")
            self.error_free = False

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    def _minimize_radDists(self, evalTime: float, crossCalib: bool = False) -> None:
        """
        Run leastsq minimization for every fit candidate stored in
        self.fits[evalTime].
        """

        # --- Only run if error free
        if not self.error_free:
            return

        try:
            if self.info is None or "scale_def" not in self.info:
                return

            # --- Data used for fitting
            data_dict = self.fitData[evalTime]

            for ii in self.fits[evalTime]:

                if not isinstance(ii, int):
                    continue

                if ii % 1_000 == 0 and self.verbose:
                    print(f"Preforming fit {ii} out of {self.info['numFits']}")

                synth_dict = self.fits[evalTime][ii]["synthetic_dict"]
                pars = self.fits[evalTime][ii]["parameters"]

                try:
                    boloNames = None
                    # --- Include bolometer names if doing a cross-calibration
                    if crossCalib and (
                        self.channel_order is not None
                        and "bolometer_order" in self.channel_order
                    ):
                        boloNames = self.channel_order["bolometer_order"]

                    self.fits[evalTime][ii]["fit"] = minimize(
                        Util_emis3D.residual,
                        pars,
                        args=(
                            data_dict,
                            synth_dict,
                            self.info["scale_def"],
                            boloNames,
                            True,  # residual = True
                        ),
                        method="leastsq",
                    )

                    self.fits[evalTime]["chiSqVec"][ii] = self.fits[evalTime][ii][
                        "fit"
                    ].chisqr.item()
                except Exception as e:
                    print(f"An error occured during the {ii} iteration: {e}")
                    self.fits[evalTime]["chiSqVec"][ii] = 1.0e19

        except Exception as e:
            print(f"An error occured while fitting: {e}")

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _rebuild_radDist(
        self,
        evalTime: float,
        locationDependence: str = "locIndependent",
        radDistNumber: int = 0,
        location=None,
        bestFit: bool = False,
    ):
        """
        Rebuild a radDist object from stored info so it can be plotted.
        """

        if bestFit:
            if self.info is not None:
                d_ = self.bestFits[evalTime]["info"]["radDists"]
                locationDependence = d_["locationdependence"]
                radDistNumber = d_["radDistNumber"]
                location = d_["location"]

        # --- Gather the radDist information
        if location is None:
            rad_ = self.data["synthetic"][locationDependence][radDistNumber]
        else:
            rad_ = self.data["synthetic"][locationDependence][location][radDistNumber]

        info = rad_.info

        # --- Create the radDist
        if info["distType"].lower() == "helical":
            radDist_ = Helical(
                config=rad_.info, startR=rad_.info["startR"], startZ=rad_.info["startZ"]
            )
        elif info["distType"].lower() == "elongatedring":
            radDist_ = ElongatedRing(config=rad_.info)
        elif info["distType"].lower() == "helicalring":
            radDist_ = HelicalRing(config=rad_.info)
        else:
            print(
                f"_rebuild_radDist() only supports Helical, ElongatedRing, or HelicalRing"
            )
            return None

        # --- Update the information
        if radDist_.info is not None:
            radDist_.info.update(info)

        # --- Update the data
        radDist_.data = dict(rad_.data)

        # --- Build the tokamak
        radDist_._build_tokamak(
            tokamakName=radDist_.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=radDist_.info["eqFileName"],
        )

        # --- Build the field line, if it is a helical distribution
        if isinstance(radDist_, Helical):
            radDist_.setFieldLine()

        return radDist_

    def _post_process_fit_arrangement(
        self, evalTime: float, crossCalib: bool = False
    ) -> None:
        """Identify the best fit and reorganise synthetic data by bolometer."""

        # --- Find the best fit
        bestFitID = np.array(self.fits[evalTime]["chiSqVec"]).argmin().item()

        # --- Print the results of the best fit
        print("\n" * 2)
        report_fit(self.fits[evalTime][bestFitID]["fit"])
        print("\n" * 2)

        # --- Store the best fit
        if not hasattr(self, "bestFits"):
            self.bestFits = {}

        self.bestFits[evalTime] = self.fits[evalTime][bestFitID]
        self.bestFits[evalTime]["bestFitID"] = bestFitID

        if self.info is None or "scale_def" not in self.info:
            print("No scale_def found in the config file")
            return

        boloNames = None
        if crossCalib and (
            self.channel_order is not None and "bolometer_order" in self.channel_order
        ):
            boloNames = self.channel_order["bolometer_order"]

        # --- First multiply the synthetic data by the fit parameters
        data_ = Util_emis3D.residual(
            self.bestFits[evalTime]["fit"].params,
            None,
            self.bestFits[evalTime]["synthetic_dict"],
            self.info["scale_def"],
            boloNames=boloNames,
            residual=False,
        )

        # --- Arrange each set in a dictionary based on the bolometer channel name
        self.bestFits[evalTime]["synthData"] = {}
        for emissionName in self.bestFits[evalTime]["synthetic_dict"]["emissionNames"]:
            self.bestFits[evalTime]["synthData"][emissionName] = {}
            temp_dict = {}
            # --- Map each data point to the correct channel
            for ii in range(len(self.channel_order["channel_list"])):
                temp_dict.update(
                    dict(
                        zip(
                            self.channel_order["channel_list"][ii],
                            data_[emissionName][ii],
                        )
                    )
                )

            # --- Loop over the bolometer and channels to build the lists
            for bolo_ in self.data["observed"]:
                self.bestFits[evalTime]["synthData"][emissionName][bolo_] = [
                    temp_dict[ch] for ch in self.data["observed"][bolo_]["channelOrder"]
                ]

        # --- Grab the radDist info and store it
        locdependence = self.fits[evalTime][bestFitID]["info"]["radDists"][
            "locationdependence"
        ]
        radDistNumber = self.fits[evalTime][bestFitID]["info"]["radDists"][
            "radDistNumber"
        ]
        loc_ = self.fits[evalTime][bestFitID]["info"]["radDists"]["location"]

        if loc_ is not None:
            self.bestFits[evalTime]["radDistInfo"] = self.data["synthetic"][
                locdependence
            ][loc_][radDistNumber].info
        else:
            self.bestFits[evalTime]["radDistInfo"] = self.data["synthetic"][
                locdependence
            ][radDistNumber].info

        # --- Rebuild the radDist
        self.bestFits[evalTime]["radDist"] = self._rebuild_radDist(
            evalTime=evalTime,
            bestFit=True,
        )

    def _post_process_radiation_distribution(self, evalTime: float) -> None:
        """Calculates the radiation amplitude distribution from the best fit"""

        if self.info is None:
            return

        scale_def = self.info["scale_def"]
        radDist_ = self.bestFits[evalTime]["radDist"]
        params = self.bestFits[evalTime]["fit"].params.valuesdict()

        mu = self.bestFits[evalTime]["synthetic_dict"]["injectionLocation_rad"]
        mu_deg = self.bestFits[evalTime]["synthetic_dict"]["injectionLocation"]
        numTransits = len(self.bestFits[evalTime]["radDistInfo"]["emissionNames"]) / 2.0

        self.bestFits[evalTime]["radiation_distribution"] = {}
        rad_distribution = self.bestFits[evalTime]["radiation_distribution"]

        if radDist_.info is not None and "distType" in radDist_.info:
            emissionNames = (
                ["clockwise", "counterClock"]
                if radDist_.info["distType"] == "helical"
                else radDist_.info["emissionNames"]
            )
        else:
            emissionNames = self.bestFits[evalTime]["synthetic_dict"]["emissionNames"]

        # --- Lists to fit the whole radiation distribution too
        x_all = []
        y_all = []

        for emissionName in emissionNames:

            # --- Fit assumed phi - mu = 0, aka center is on the injection location
            dphi = np.linspace(-np.pi, np.pi, 200)
            rad_distribution[emissionName] = {}

            a = params[f"a_{mu_deg}"]
            b = params[f"b_{emissionName}_{mu_deg}"]

            scale_ = Util_emis3D.scale_wrapper(
                a=a,
                b=b,
                phi=np.zeros(1),
                dphi=dphi,
                mu=0.0,
                scale_def=scale_def,
                emissionName=emissionName,
            )

            if "clockwise" in emissionName:
                loc_ = dphi <= 0
                # Scale back up to 2 pi since the helical distribution is a full revolutions
                dphi_scale = 2.0 * numTransits
            elif "counterClock" in emissionName:
                loc_ = dphi > 0
                dphi_scale = 2.0 * numTransits
            else:
                loc_ = np.full(dphi.shape[0], fill_value=True)
                dphi_scale = 1.0

            dphi *= dphi_scale

            phi_unwrapped = dphi[loc_] + mu
            amplitude_ = scale_[loc_]

            # --- Add them to the total distribution
            x_all.extend(phi_unwrapped)
            y_all.extend(amplitude_)

            # --- Wrap phi so it is from 0 to 2pi
            phi_wrapped = np.mod(phi_unwrapped, 2.0 * np.pi)
            sort_indx = np.argsort(phi_wrapped)

            # --- Populate the arrays
            rad_distribution[emissionName]["phi"] = phi_wrapped[sort_indx]
            rad_distribution[emissionName]["amplitude"] = amplitude_[sort_indx]
            rad_distribution[emissionName]["phi_unwrapped"] = phi_unwrapped
            rad_distribution[emissionName]["amplitude_unwrapped"] = amplitude_
            rad_distribution[emissionName]["phi_left_handed_deg"] = np.rad2deg(
                2.0 * np.pi - phi_wrapped[sort_indx]
            )

        rad_distribution["total"] = {"phi": x_all, "amp": y_all}

    def _post_process_calculations(self, evalTime: float) -> None:
        """
        Rebuild radDist, calculate powerPerBin, and compute the toroidal
        peaking factor (TPF).
        """

        # --- Only run if self._post_process_fit_arrangement() has been run
        if not hasattr(self, "bestFits"):
            print(
                "Please run self._post_process_fit_arrangement() before "
                "self._post_process_calculations()"
            )
            return

        # --- Shorten some variable calls
        radDist_ = self.bestFits[evalTime]["radDist"]
        rad_distribution = self.bestFits[evalTime]["radiation_distribution"]
        emissionNames = radDist_.info["emissionNames"]

        self.bestFits[evalTime]["powerPerBin"] = {}
        powerPerBin = self.bestFits[evalTime]["powerPerBin"]

        mu = self.bestFits[evalTime]["synthetic_dict"]["injectionLocation_rad"]

        x_all = []
        y_all = []

        # Use the first emission name once to get the synthetic scale factor
        # (it is the same for all emissionNames).
        first_emission = emissionNames[0]
        scale_synth = self.bestFits[evalTime]["synthetic_dict"][first_emission][
            "scaleSynth"
        ]

        # --- Unwrap and combine the powerPerBin for both radiation distribution functions
        for emissionName in emissionNames:
            powerPerBin[emissionName] = {}
            phibin_center = radDist_.data["toroidalRadiatedPower"][emissionName][
                "phi_array"
            ]
            ppb_amp = radDist_.data["toroidalRadiatedPower"][emissionName]["P_pol"]

            # --- elongatedRing distributions are symmetric, so we can skip a lot of this
            if emissionName == "elongatedRing":
                # Symmetric: extend with a constant to avoid interpolation artefacts
                y_ = np.mean(ppb_amp)
                x_all.extend(np.linspace(0, 2.0 * np.pi, 20))
                y_all.extend(np.full(20, fill_value=y_))
            else:
                # --- Unwrap the powerPerBin
                dphi = phibin_center - mu

                # --- Determine the offset
                offset = 0.0
                if "rev" in emissionName:
                    offset = int(emissionName[-1]) * 2.0 * np.pi

                # --- Remove the data point at dphi = 0, it is always wrong for some reason
                indx = np.abs(dphi).argmin()
                dphi = np.delete(dphi, indx)
                ppb_amp = np.delete(ppb_amp, indx)

                if "counterClock" in emissionName:
                    dphi[dphi <= 0] += 2.0 * np.pi
                elif "clockwise" in emissionName:
                    dphi[dphi > 0] -= 2.0 * np.pi

                # --- Arrange the data in ascending order
                sort_ = np.argsort(np.array(dphi))
                dphi_ = dphi[sort_] + mu + offset

                # --- Clockwise data should be negative
                if "clockwise" in emissionName:
                    dphi_ *= -1.0

                # --- Add mu back, then put it in the master array
                x_all.extend(dphi_)
                y_all.extend(ppb_amp[sort_])

        # --- Arrange the data in ascending order
        sort_ = np.argsort(np.array(x_all))
        x_all_ppb = np.array(x_all)[sort_]
        y_all_ppb = np.array(y_all)[sort_]

        # --- Perform fits on the data
        x_min, x_max = x_all_ppb[0], x_all_ppb[-1]
        phi_ = np.linspace(x_min, x_max, 500)
        ppb_fit = np.interp(phi_, x_all_ppb, y_all_ppb)

        # --- Fit the radiation distribution
        x_pts = np.array(rad_distribution["total"]["phi"].copy())
        y_pts = np.array(rad_distribution["total"]["amp"].copy())

        # --- Wrap the distribution if it is only from -pi to pi
        if np.max(x_pts) - np.min(x_pts) == 2.0 * np.pi:
            x_pts = x_pts % (2.0 * np.pi)
            s_ = np.argsort(x_pts)
            x_pts = x_pts[s_]
            y_pts = y_pts[s_]

        y_rad_distr = np.interp(phi_, x_pts, y_pts)
        ppb_total = scale_synth * ppb_fit * y_rad_distr

        # --- Wrap back to [0, 360]
        phi_wrapped = np.linspace(0, 2.0 * np.pi, 360)
        ppb_total_wrapped = np.zeros(phi_wrapped.shape[0])

        # --- Find all the equivilent data within the range
        for ii, theta in enumerate(phi_wrapped):
            ks = np.arange(
                (x_min - theta) // (2.0 * np.pi),
                (x_max - theta) // (2.0 * np.pi) + 1,
            )
            x_vals = theta + 2.0 * np.pi * ks

            # --- Keep only the values within the range
            x_vals = x_vals[(x_vals >= x_min) & (x_vals <= x_max)]

            ppb_total_wrapped[ii] = np.sum(np.interp(x_vals, phi_, ppb_total))

        # --- Something funky happens with the elongatedRing distribution when they
        # are wrapped, so just keep the unwrapped version
        is_elongated = emissionNames == ["elongatedRing"] or (
            len(emissionNames) == 1 and emissionNames[0] == "elongatedRing"
        )
        if is_elongated:
            phi_wrapped = phi_
            ppb_total_wrapped = ppb_total

        # --- Store the results
        powerPerBin["total"] = {
            "phi_unwrapped": phi_,
            "powerPerBin_unwrapped": ppb_total,
            "phi": phi_wrapped,
            "powerPerBin": ppb_total_wrapped,
            "toroidal_peaking_factor": (
                np.max(ppb_total_wrapped)
                / (simpson(ppb_total_wrapped, x=phi_wrapped) / (2.0 * np.pi))
            ),
        }

    # ------------------------------------------------------------------
    # Cleanup / persistence
    # ------------------------------------------------------------------

    def _cleanup_fits(self, evalTime: float) -> None:
        """Delete non-best-fit entries from self.fits to reclaim memory."""

        if self.verbose:
            print(f"Deleting bad fits for = {evalTime:.2f} ms")

        # --- Delete the bad fits to save memory
        if hasattr(self, "fits") and evalTime in self.fits:
            best_id = self.bestFits[evalTime]["bestFitID"]
            for ii in list(self.fits[evalTime].keys()):
                if isinstance(ii, int) and ii != best_id:
                    del self.fits[evalTime][ii]

    def _save_bestFits(self) -> None:
        """Serialise bestFits and fitData to a dill file."""

        def save_results(filename, data):
            with open(filename, "wb") as f:
                dill.dump(data, f)

        if (
            self.info is None
            or "shot" not in self.info
            or "tokamakName" not in self.info
        ):
            return

        keys = list(self.bestFits.keys())
        if len(keys) > 1:
            t_min = np.min(keys)
            t_max = np.max(keys)
            filename = f"{self.info['shot']}_bestFits_{t_min:.3f}_to_{t_max:.3f}.dill"
        else:
            filename = f"{self.info['shot']}_bestFits_{keys[0]:.3f}.dill"

        pathFileName = join(
            EMIS3D_INPUTS_DIRECTORY,
            self.info["tokamakName"],
            "runs",
            str(self.info["shot"]),
            filename,
        )

        save_results(
            pathFileName, {"fit_data": self.fitData, "bestFits": self.bestFits}
        )
        print(f"Best fits and fitData saved to: {pathFileName}")

    def _load_bestFits(self, path: str = "") -> None:
        """Load bestFits and fitData from a previously saved dill file."""

        def load_results(filename):
            with open(filename, "rb") as f:
                return dill.load(f)

        self.bestFits = {}
        self.fitData = {}

        temp = load_results(path)
        if isinstance(temp, dict):
            for key in list(temp.keys()):
                for evalTime in temp[key]:
                    if key == "fit_data":
                        self.fitData[evalTime] = temp[key][evalTime]
                    elif key == "bestFits":
                        self.bestFits[evalTime] = temp[key][evalTime]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_bestFit(self, evalTime: float, save: bool = False) -> None:
        """Plots the fit synthetic signal, data, and radDist for the given evalTime."""
        print(f"Plotting the best fit")

        if self.info is None:
            return

        # --- Rebuild the bestFit radDist
        if not hasattr(self, "bestFits"):
            print(
                "Please run _post_process_fit_arrangement() and "
                "_post_process_calculations() before _plot_bestFit()"
            )
            return

        # --- Find the eqFileName, if it exists
        eqFileName = self.bestFits[evalTime]["radDistInfo"]["eqFileName"]

        tok = Tokamak(
            tokamakName=self.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=eqFileName,
            loadBolometers=True,
        )

        bolometers = self.info["BOLOMETERS"]
        units = self.bestFits[evalTime]["radDist"].info["units"]
        units_label = {
            "Power": "[W]",
            "Radiance": "[W / (m2 sr)]",
            "Brightness": "[W / m2]",
        }.get(units, "[arb]")

        num_columns = len(bolometers) + 1
        f = plt.figure(figsize=(15, 8))

        # --- Plot the bolometer chords and radDist contour
        count_ = 0
        for boloName in bolometers:
            count_ += 1
            ax = f.add_subplot(2, num_columns, count_)
            tok._plot_first_wall(ax)
            for bolo_ in tok.bolometers:
                tok._plot_bolometers(ax, boloName)

                # --- Add the radDist plot
                phi = bolo_.info["CAMERA_POSITION_R_Z_PHI"][2]
                self.bestFits[evalTime]["radDist"].plotCrossSection(
                    phi=np.deg2rad(phi), ax=ax
                )
            ax.set_title(boloName)

        # --- Plot the contour at the injection location
        count_ += 1
        ax = f.add_subplot(2, num_columns, count_)
        tok._plot_first_wall(ax)
        phi = self.info.get("injectionLocation", 0)
        self.bestFits[evalTime]["radDist"].plotCrossSection(phi=np.deg2rad(phi), ax=ax)
        ax.set_title(f"Injection location = {phi:.2f} degrees")

        # --- Plot the observed emissivities
        colors = ["green", "orangered", "blue", "cyan", "magenta"]
        markers = ["^", "o", "s", "D", "v"]

        for ii, bolo_ in enumerate(bolometers):
            count_ += 1
            ax = f.add_subplot(2, num_columns, count_)
            numChan = len(self.fitData[evalTime]["boloData"][bolo_])
            channels = np.arange(1, numChan + 1, 1)

            # --- Observed data
            ax.errorbar(
                channels,
                self.fitData[evalTime]["boloData"][bolo_],
                yerr=self.fitData[evalTime]["boloData_error"][bolo_],
                marker="s",
                ms=5,
                c="black",
                linestyle="none",
                label="data",
            )

            tot_emission = np.zeros(numChan)
            for jj, emissionName in enumerate(self.bestFits[evalTime]["synthData"]):
                em_data = np.array(
                    self.bestFits[evalTime]["synthData"][emissionName][bolo_]
                )
                tot_emission += em_data

                ax.plot(
                    channels,
                    em_data,
                    marker=markers[jj % len(markers)],
                    color=colors[jj % len(colors)],
                    label=f"{emissionName} emission",
                )

            ax.plot(
                channels,
                tot_emission,
                color="purple",
                label="total emission",
            )

            ax.set_xlabel("Channel Number")
            ax.set_ylabel(f"Emission {units_label}")
            ax.set_title(f"{bolo_}")
            ax.set_ylim(0, ax.get_ylim()[1])

            if ii == 0:
                ax.legend(fontsize=8)

        # --- Plot the radiation behavior
        tpf_ax = f.add_subplot(2, num_columns, count_ + 1)
        y_data = self.bestFits[evalTime]["powerPerBin"]["total"]["powerPerBin"]
        scale = np.floor(np.log10(np.nanmax(y_data)))
        tpf_ax.plot(
            np.rad2deg(self.bestFits[evalTime]["powerPerBin"]["total"]["phi"]),
            y_data / 10**scale,
            color="black",
            linewidth=2.0,
        )
        tpf_ax.set_ylim(
            np.floor(np.nanmin(y_data / 10**scale)),
            np.ceil(np.nanmax(y_data / 10**scale)),
        )
        tpf_ax.set_xlabel("phi [degrees]")
        tpf_ax.set_ylabel(f"radiation [$10^{{{int(scale)}}}$ arb]")
        tpf_ax.set_title(
            f"time = {evalTime:.2f} ms, "
            f"TPF: {self.bestFits[float(evalTime)]['powerPerBin']['total']['toroidal_peaking_factor']:.2f}"
        )

        plt.tight_layout()

        if save:
            if "shot" in self.info and "tokamakName" in self.info:
                filename = f"{self.info['shot']}_{evalTime:.3f}.png"
                img_dir = join(
                    EMIS3D_INPUTS_DIRECTORY,
                    self.info["tokamakName"],
                    "runs",
                    str(self.info["shot"]),
                    "images",
                )

                # --- Make the directory
                os.makedirs(img_dir, exist_ok=True)
                out_path = join(img_dir, filename)
                plt.savefig(out_path, dpi=100, format="png")
                print(f"Figure saved to {out_path}")

        else:
            plt.show()

    # ------------------------------------------------------------------
    # Placeholder / future work
    # ------------------------------------------------------------------

    def _combine_synthetics_for_fits(self, evalTime: float) -> None:
        """
        Combines radDists from up to two injection locations iteravely

        """

        pass

        """
        Do something like this:

        def combine_dicts(a, b=None):
            c = {}
            counter = 0

            if b is None:
                # Only loop over a
                for a_key in a.keys():
                    c[counter] = {
                        "index a": a_key,
                        "data a": a[a_key]["data"]
                    }
                    counter += 1
            else:
                # Loop over all permutations of a and b
                for a_key, b_key in itertools.product(a.keys(), b.keys()):
                    c[counter] = {
                        "index a": a_key,
                        "index b": b_key,
                        "data a": a[a_key]["data"],
                        "data b": b[b_key]["data"]
                    }
                    counter += 1
            return c
        """

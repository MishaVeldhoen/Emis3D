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


Biggest Issues:
1. Find good radDist functions that represent what is going on
    - Add a tomography radDist mapping function (like BOLT?)


BUG:
1. Is the total radiated power calculated correctly?

"""

import os
import logging
import time
import warnings

import dill
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, report_fit
from scipy.integrate import simpson

import main.Util_emis3D as Util_emis3D

logger = logging.getLogger(__name__)
from main.Globals import EMIS3D_INPUTS_DIRECTORY
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

        # Configure logging level for the whole 'main' package
        if verbose:
            logging.getLogger("main").setLevel(logging.DEBUG)

        if initialize:
            self._initialize(tokamakName=tokamakName, runConfigName=runConfigName)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(
        self, tokamakName: str | None = None, runConfigName: str | None = None
    ) -> None:
        """Load configuration, bolometer data and radDists."""
        if tokamakName is None or runConfigName is None:
            raise ValueError(
                "tokamakName and runConfigName are required to initialise Emis3D"
            )

        Util_emis3D.print_intro()
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
        self._prepare_fits(evalTime=evalTime, crossCalib=crossCalib)
        t_start = time.time()
        self._minimize_radDists(evalTime=evalTime)
        logger.info("Fitting done in %.2f seconds", time.time() - t_start)
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

        if pathFileName is None:
            if tokamakName is not None and runConfigName is not None:
                pathFileName = str(
                    EMIS3D_INPUTS_DIRECTORY / tokamakName / "runs" / runConfigName
                )
            else:
                raise ValueError(
                    "tokamakName, runConfigName, or pathFileName is None; "
                    "cannot load config file"
                )

        if not os.path.isfile(str(pathFileName)):
            raise FileNotFoundError(f"Config file not found: {pathFileName}")

        self.info = config_loader(str(pathFileName), verbose=self.verbose)

        if self.info is None:
            raise RuntimeError(f"Could not load the configuration file: {pathFileName}")

        self.info["tokamakName"] = tokamakName
        self.info["runConfigName"] = runConfigName

    def _load_radDists(self) -> None:
        """
        Loads each radDist within the directories listed in the config file:
        radDistDirectories_LocIndependent, and radDistDirectories_LocDependent
        """

        if self.verbose:
            logger.debug("Loading radDists")

        # --- Initialise synthetic signal arrays
        self.data.update({"synthetic": {"locDependent": {}, "locIndependent": {}}})
        self.files = []

        if self.info is None or (
            "radDistDirectories_LocIndependent" not in self.info
            and "radDistDirectories_LocDependent" not in self.info
        ):
            raise RuntimeError("No radDistDirectories found in the config file")

        # --- Load location independent radDists
        count_ = 0
        for dir_ in self.info["radDistDirectories_LocIndependent"]:
            pathFileName = os.path.join(
                EMIS3D_INPUTS_DIRECTORY,
                self.info["tokamakName"],
                "radDists",
                dir_,
            )
            for file_ in get_filenames_in_directory(pathFileName):
                try:
                    temp_ = RadDistFitting(radDistPath=file_)
                    self.data["synthetic"]["locIndependent"][count_] = temp_
                    count_ += 1
                except Exception as e:
                    warnings.warn(f"Skipping radDist {file_}: {e}", stacklevel=2)
                    logger.warning("Skipping radDist %s: %s", file_, e)

        # --- Load location dependent radDists
        for ii, dir_ in enumerate(self.info["radDistDirectories_LocDependent"]):
            self.data["synthetic"]["locDependent"][f"loc_{ii}"] = {}
            pathFileName = os.path.join(
                EMIS3D_INPUTS_DIRECTORY,
                self.info["tokamakName"],
                "radDists",
                dir_,
            )
            count_ = 0
            for file_ in get_filenames_in_directory(pathFileName):
                try:
                    temp_ = RadDistFitting(radDistPath=file_)
                    self.data["synthetic"]["locDependent"][f"loc_{ii}"][count_] = temp_
                    count_ += 1
                except Exception as e:
                    warnings.warn(f"Skipping radDist {file_}: {e}", stacklevel=2)
                    logger.warning("Skipping radDist %s: %s", file_, e)

        if self.verbose:
            logger.debug("Done loading radDists")

    def _load_bolometer_data(self) -> None:
        """
        Load pre-calibrated SXR/bolometer data from ``inputs/{tokamakName}/sxrData/``.
        The filename is taken from ``dataFileName`` in the run config.
        """

        if self.info is None or "BOLOMETERS" not in self.info:
            raise RuntimeError("No BOLOMETERS found in the config file")

        self.data["observed"] = {}
        for bolo_ in self.info["BOLOMETERS"]:
            pathFileName = os.path.join(
                EMIS3D_INPUTS_DIRECTORY,
                self.info["tokamakName"],
                "sxrData",
                self.info["BOLOMETERS"][bolo_]["dataFileName"],
            )
            if not os.path.isfile(pathFileName):
                raise FileNotFoundError(
                    f"Bolometer data file not found: {pathFileName}"
                )

            logger.debug("Loading bolometer data: %s", pathFileName)

            temp_ = read_h5(pathFileName)

            for ii, ch_ in enumerate(temp_["channelOrder"]):
                temp_["channelOrder"][ii] = ch_.decode("utf-8")

            temp_["DATA_CALIBRATED"] *= self.info["BOLOMETERS"][bolo_].get(
                "scalingFactor", 1.0
            )

            self.data["observed"][bolo_] = temp_

    def _create_master_channel_order(self) -> None:
        """
        Build a master channel list from the loaded bolometer data.
        All radDists and raw data will be ordered to match this list.

        List is of the format:
        [[bolo1_1, bolo1_2, ...], [bolo2_1, bolo2_2, ...], ...]
        """

        if self.info is None or "BOLOMETERS" not in self.info:
            raise RuntimeError("No BOLOMETERS found in the config file")

        self.channel_order = {
            "bolometer_order": [],
            "channel_list": [],
        }

        for bolo in self.data["observed"]:
            self.channel_order["bolometer_order"].append(bolo)
            self.channel_order["channel_list"].append(
                list(self.data["observed"][bolo]["channelOrder"])
            )

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

        # --- Average the data, include endpoint to prevent errors when dt is too small
        vals = np.mean(
            self.data["observed"][arrayName]["DATA_CALIBRATED"][:, start : end + 1],
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

        if not hasattr(self, "fitData"):
            self.fitData = {}

        # NOTE:
        """
        - Observed is what is used in the minimization process, it is a set of
          nested list, the same way that self.channel_order['channel_list'] is organized
        - Bolo data is organzied to make plotting easier since it is a dict of each bolometer.
        """

        self.fitData[evalTime] = {
            "observed": [],
            "observed_error": [],
            "boloData": {},
            "boloData_error": {},
        }

        # --- Average and map the data to a dict
        data_ = {}
        for bolo_ in self.data["observed"]:
            temp = self._average_observed_data(arrayName=bolo_, evalTime=evalTime)
            data_.update(dict(zip(self.data["observed"][bolo_]["channelOrder"], temp)))

        self.fitData[evalTime]["dataMap"] = data_

        # --- Find the max value in all the channels
        max_ = max(
            (data_[ch] for bolo_ in self.channel_order["channel_list"] for ch in bolo_),
            default=0.0,
        )

        for ii, bolo_chans in enumerate(self.channel_order["channel_list"]):
            temp = []
            temp_e = []

            # --- Find the max value for that specific array
            """
            max_ = 0
            for channel in channels:
                if data_[channel] > max_:
                    max_ = data_[channel]
            """

            for channel in bolo_chans:
                val = data_[channel]
                temp.append(val)

                # --- Large error for zero signal, use definition otherwise

                if val > 1.0:
                    err_frac = 0.05
                    if self.info is not None:
                        err_frac = Util_emis3D.signal_error(
                            self.info["error_definition"], val, max_, scale_factor=1.0
                        )
                    err_ = val * err_frac
                else:
                    err_ = np.float64(1.0e4)

                temp_e.append(err_)

            self.fitData[evalTime]["observed"].append(temp)
            self.fitData[evalTime]["observed_error"].append(temp_e)

            # --- Temp is a nested list of bolometers [[Bolo1-1, Bolo1-2, ...], [Bolo2-1, Bolo2-2], etc.]
            bolo_name = self.channel_order["bolometer_order"][ii]
            self.fitData[evalTime]["boloData"][bolo_name] = temp
            self.fitData[evalTime]["boloData_error"][bolo_name] = temp_e

        if self.verbose:
            logger.debug("→ Observed data prepared for fitting")

    def _prepare_synthetic_for_fits(
        self, evalTime: float, crossCalib: bool = False
    ) -> None:
        """Prepare and parameterise every radDist for the minimisation."""

        print_minor_error = True

        def print_error(radD):
            logger.info("")
            logger.warning("-" * 10 + "MINOR ERROR" + "-" * 10)
            logger.warning(
                f"Channels: {radD.info['ERROR CHANNELS']}were not found in the radDist, they will be ignored"
            )
            logger.debug("-" * 31)
            logger.info("")

        logger.debug("→ Preparing synthetic data for fitting")

        # --- Scale the synthetic data to observed for better fitting
        max_data_val = find_max_nested_lists(self.fitData[evalTime]["observed"])

        if self.info is not None and "enable_dphi_scaling" in self.info:

            boloNames = (
                self.channel_order["bolometer_order"]
                if self.info["enable_dphi_scaling"]
                else None
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
                        enable_dphi_scaling=self.info["enable_dphi_scaling"],
                    )
                    if "ERROR CHANNELS" in radD.info and print_minor_error:
                        print_error(radD)
                        print_minor_error = False

            # --- Arrange and create parameters for the location independent data
            for number_ in self.data["synthetic"]["locIndependent"]:
                radD = self.data["synthetic"]["locIndependent"][number_]
                radD.prepare_for_fits(
                    self.channel_order["channel_list"], data_max=max_data_val
                )
                radD.create_parameters(
                    boloNames=boloNames,
                    enable_dphi_scaling=self.info["enable_dphi_scaling"],
                )

                if "ERROR CHANNELS" in radD.info and print_minor_error:
                    print_error(radD)
                    print_minor_error = False

        if self.verbose:
            logger.debug("→ Done preparing synthetic data for fit")

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

        in_ = {}
        if radDist_.info is not None:
            in_ = radDist_.info

        synthetic_dict = {
            "info": in_,
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

        if self.verbose:
            logger.debug("→ Preparing data for fitting")

        self._prepare_data_for_fit(evalTime=evalTime)
        self._prepare_synthetic_for_fits(evalTime=evalTime, crossCalib=crossCalib)

        if self.verbose:
            logger.debug("→ Arranging radDists for fitting")

        if not hasattr(self, "fits"):
            self.fits = {}

        # --- Initialize the fitting dictionary
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

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    def _minimize_radDists(self, evalTime: float, crossCalib: bool = False) -> None:
        """
        Run leastsq minimization for every fit candidate stored in
        self.fits[evalTime].
        """

        try:
            if self.info is None or "scale_def" not in self.info:
                return

            # --- Data used for fitting
            data_dict = self.fitData[evalTime]

            for ii in self.fits[evalTime]:

                if not isinstance(ii, int):
                    continue

                if ii % 1_000 == 0 and self.verbose:
                    logger.debug(f"→ Preforming fit {ii} out of {self.info['numFits']}")

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
                    ].redchi
                except Exception as e:
                    logger.warning(f"An error occurred during the {ii} iteration: {e}")
                    self.fits[evalTime]["chiSqVec"][ii] = 1.0e19

        except Exception as e:
            logger.warning(f"An error occurred while fitting: {e}")

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
            logger.info(
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
        logger.info("")
        report_fit(self.fits[evalTime][bestFitID]["fit"])
        logger.info("")

        # --- Store the best fit
        if not hasattr(self, "bestFits"):
            self.bestFits = {}

        self.bestFits[evalTime] = self.fits[evalTime][bestFitID]
        self.bestFits[evalTime]["bestFitID"] = bestFitID

        if self.info is None or "scale_def" not in self.info:
            logger.info("No scale_def found in the config file")
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
        params = self.bestFits[evalTime]["fit"].params.valuesdict()

        mu = self.bestFits[evalTime]["synthetic_dict"]["injectionLocation_rad"]
        mu_deg = self.bestFits[evalTime]["synthetic_dict"]["injectionLocation"]
        numTransits = len(self.bestFits[evalTime]["radDistInfo"]["emissionNames"]) / 2.0

        self.bestFits[evalTime]["radiation_distribution"] = {}
        rad_distribution = self.bestFits[evalTime]["radiation_distribution"]

        emissionNames = self.bestFits[evalTime]["synthetic_dict"]["emissionNames"]

        # --- Lists to fit the whole radiation distribution too
        # Shared output grid: clean [0, 2π] at 500 points.
        # All components are summed here so rad_distribution["total"] needs no
        # further wrapping in _post_process_calculations.
        phi_grid = np.linspace(0, 2.0 * np.pi, 500)
        amp_total = np.zeros_like(phi_grid)

        for emissionName in emissionNames:

            # --- Fit assumed phi - mu = 0, aka center is on the injection location
            dphi = np.linspace(-np.pi, np.pi, 500)
            rad_distribution[emissionName] = {}

            # Both directions share amplitude 'a'; their individual decay constant
            # 'b' controls how fast each falls off away from the injection location.
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
                # CW half:  [-π, 0]  → [-2π·N, 0], for helicals
                loc_ = (dphi > -np.pi) & (
                    dphi <= 0
                )  # CW:  exclude the "arrival" endpoint
                # Scale back up to 2 pi since the helical distribution is a full revolutions
                dphi_scale = 2.0 * numTransits
            elif "counterClock" in emissionName:
                # CCW half: (0,  π]  → (0, 2π·N], for helicals
                loc_ = (dphi > 0) & (
                    dphi < np.pi
                )  # CCW: exclude the "arrival" endpoint
                dphi_scale = 2.0 * numTransits
            else:
                loc_ = np.full(dphi.shape[0], fill_value=True)
                dphi_scale = 1.0

            phi_unwrapped = dphi[loc_] * dphi_scale + mu
            amplitude_ = scale_[loc_]

            # Sort the unwrapped domain once for use in both the alias loop and
            # the per-emission wrapped storage below.
            sort_uw = np.argsort(phi_unwrapped)
            phi_uw_sorted = phi_unwrapped[sort_uw]
            amp_uw_sorted = amplitude_[sort_uw]
            x_min_em = phi_uw_sorted[0]
            x_max_em = phi_uw_sorted[-1]

            # --- Alias summation onto the shared [0, 2π] grid -------------------
            # For numTransits = 1 each angle has exactly one alias per component.
            # For numTransits = N > 1 there are N aliases and they all contribute.
            for ii, theta in enumerate(phi_grid):
                k_lo = int(np.ceil((x_min_em - theta) / (2.0 * np.pi)))
                k_hi = int(np.floor((x_max_em - theta) / (2.0 * np.pi)))
                if k_lo > k_hi:
                    continue
                x_aliases = theta + 2.0 * np.pi * np.arange(k_lo, k_hi + 1)
                x_aliases = x_aliases[(x_aliases >= x_min_em) & (x_aliases <= x_max_em)]
                if x_aliases.size > 0:
                    amp_total[ii] += np.sum(
                        np.interp(x_aliases, phi_uw_sorted, amp_uw_sorted)
                    )
            # ---------------------------------------------------------------------

            # Per-emission diagnostics (individual plotting, left-handed angle, etc.)
            phi_wrapped_em = np.mod(phi_unwrapped, 2.0 * np.pi)
            sort_w = np.argsort(phi_wrapped_em)

            rad_distribution[emissionName]["phi"] = phi_wrapped_em[sort_w]
            rad_distribution[emissionName]["amplitude"] = amplitude_[sort_w]
            rad_distribution[emissionName]["phi_unwrapped"] = phi_unwrapped
            rad_distribution[emissionName]["amplitude_unwrapped"] = amplitude_
            rad_distribution[emissionName]["phi_left_handed_deg"] = np.rad2deg(
                2.0 * np.pi - phi_wrapped_em[sort_w]
            )

        # Already a clean, monotone [0, 2π] grid — no wrapping needed downstream.
        rad_distribution["total"] = {"phi": phi_grid, "amp": amp_total}

    def _post_process_calculations(self, evalTime: float) -> None:
        """
        Rebuild radDist, calculate powerPerBin, and compute the toroidal
        peaking factor (TPF).
        """

        # --- Only run if self._post_process_fit_arrangement() has been run
        if not hasattr(self, "bestFits"):
            logger.info(
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
                # Symmetric: extend with a constant to avoid interpolation artifacts
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

                # BUG: Here? CW negative flipped to negative twice?
                if "counterClock" in emissionName:
                    dphi[dphi <= 0] += 2.0 * np.pi
                elif "clockwise" in emissionName:
                    dphi[dphi > 0] -= 2.0 * np.pi

                # --- Arrange the data in ascending order
                sort_ = np.argsort(dphi)
                # dphi_ = dphi[sort_] + mu + offset

                # --- Clockwise data should be negative
                if "clockwise" in emissionName:
                    dphi_ = -dphi[sort_] + mu + offset
                    # OLD: dpi_ *= -1
                else:
                    dphi_ = dphi[sort_] + mu + offset

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
        x_pts = np.array(rad_distribution["total"]["phi"])
        y_pts = np.array(rad_distribution["total"]["amp"])
        y_rad_distr = np.interp(phi_ % (2.0 * np.pi), x_pts, y_pts)
        ppb_total = scale_synth * ppb_fit * y_rad_distr

        # Wrap back to [0, 2π]
        phi_wrapped = np.linspace(0, 2.0 * np.pi, 360)
        ppb_total_wrapped = np.zeros(phi_wrapped.shape[0])

        for ii, theta in enumerate(phi_wrapped):
            ks = np.arange(
                (x_min - theta) // (2.0 * np.pi),
                (x_max - theta) // (2.0 * np.pi) + 1,
            )
            x_vals = theta + 2.0 * np.pi * ks
            x_vals = x_vals[(x_vals >= x_min) & (x_vals <= x_max)]
            ppb_total_wrapped[ii] = np.sum(np.interp(x_vals, phi_, ppb_total))

        is_elongated = emissionNames == ["elongatedRing"] or (
            len(emissionNames) == 1 and emissionNames[0] == "elongatedRing"
        )
        if is_elongated:
            phi_wrapped = phi_
            ppb_total_wrapped = ppb_total

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

        """
        # OLD
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
        """

    # ------------------------------------------------------------------
    # Cleanup / persistence
    # ------------------------------------------------------------------

    def _cleanup_fits(self, evalTime: float) -> None:
        """Delete non-best-fit entries from self.fits to reclaim memory."""

        if self.verbose:
            logger.debug(f"→ Deleting bad fits for = {evalTime:.4f} ms")

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

        pathFileName = (
            EMIS3D_INPUTS_DIRECTORY
            / self.info["tokamakName"]
            / "runs"
            / str(self.info["shot"])
            / filename
        )

        save_results(
            pathFileName, {"fit_data": self.fitData, "bestFits": self.bestFits}
        )
        logger.debug(f"→ Best fits and fitData saved to: {pathFileName}")

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
        logger.debug(f"→ Plotting the best fit")

        if self.info is None:
            return

        # --- Rebuild the bestFit radDist
        if not hasattr(self, "bestFits"):
            logger.info(
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
            ymax = np.nanmax(
                self.fitData[evalTime]["boloData"][bolo_]
                + np.abs(self.fitData[evalTime]["boloData_error"][bolo_])
            )
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
            if np.nanmax(tot_emission) > ymax:
                ymax = np.nanmax(tot_emission)

            ax.set_xlabel("Channel Number")
            ax.set_ylabel(f"Emission {units_label}")
            ax.set_title(f"{bolo_}")
            ax.set_ylim(0, float(ymax) * 1.02)

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
            f"time = {evalTime:.4f} ms, "
            f"TPF: {self.bestFits[float(evalTime)]['powerPerBin']['total']['toroidal_peaking_factor']:.2f}"
        )

        plt.tight_layout()

        if save:
            if "shot" in self.info and "tokamakName" in self.info:
                filename = f"{self.info['shot']}_{evalTime:.4f}.png"
                img_dir = (
                    EMIS3D_INPUTS_DIRECTORY
                    / self.info["tokamakName"]
                    / "runs"
                    / str(self.info["shot"])
                    / "images"
                )

                # --- Make the directory
                os.makedirs(img_dir, exist_ok=True)
                out_path = img_dir / filename
                plt.savefig(out_path, dpi=100, format="png")
                logger.debug(f"→ Figure saved to {out_path}")

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

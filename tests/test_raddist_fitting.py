"""
tests/test_raddist_fitting.py
=============================
Unit tests for main/radDistFitting.py (data mapping / parameter creation)
and main/Globals.py (path constants).
No cherab, raysect, or real file I/O required beyond what we fabricate in-memory.
"""

import json
import os

import numpy as np
import pytest

from main.Globals import (
    FILE_PATH,
    EMIS3D_PARENT_DIRECTORY,
    EMIS3D_TOKMAK_DIRECTORY,
    SUPPORTED_TOKAMAKS,
)
from main.radDistFitting import RadDistFitting

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------


class TestGlobals:

    def test_file_path_is_directory(self):
        assert os.path.isdir(FILE_PATH), f"FILE_PATH '{FILE_PATH}' is not a directory"

    def test_parent_directory_exists(self):
        assert os.path.isdir(EMIS3D_PARENT_DIRECTORY)

    def test_tokamak_directory_exists(self):
        assert os.path.isdir(EMIS3D_TOKMAK_DIRECTORY)

    def test_supported_tokamaks_is_list(self):
        assert isinstance(SUPPORTED_TOKAMAKS, list)
        assert len(SUPPORTED_TOKAMAKS) > 0

    def test_diii_d_is_supported(self):
        assert "DIII-D" in SUPPORTED_TOKAMAKS


# ---------------------------------------------------------------------------
# RadDistFitting (no real data; we construct a minimal JSON fixture)
# ---------------------------------------------------------------------------


def _minimal_raddist_json(
    tmp_path, units="Power", distType="elongatedRing", n_channels=4, n_bolos=2
):
    """
    Write a minimal radDist JSON file that RadDistFitting can load without
    needing cherab or actual measurement files.
    """
    emission = "elongatedRing"
    channel_tags = [f"CH{i+1:02d}" for i in range(n_channels)]

    bolo_names = [f"BOLO_{b}" for b in range(n_bolos)]
    channel_order = {}
    power_data = {}
    power_error = {}
    scale_factor = {}

    for bolo in bolo_names:
        channel_order[bolo] = channel_tags
        power_data[bolo] = [float(i + 1) for i in range(n_channels)]
        power_error[bolo] = [0.1 * (i + 1) for i in range(n_channels)]
        scale_factor[bolo] = [0.5 * (i + 1) for i in range(n_channels)]

    data = {
        "info": {
            "distType": distType,
            "emissionNames": [emission],
            "injectionLocation": 120.0,
            "tokamakName": "DIII-D",
            "units": units,
            "polSigma": 0.1,
            "elongation": 1.5,
            "rotationAngle": 0.0,
            "startR": 2.0,
            "startZ": 0.0,
            "eqFileName": None,
            "saveRunsDirectoryName": "test",
        },
        "data": {
            units: {
                "channelOrder": channel_order,
                emission: {bolo: power_data[bolo] for bolo in bolo_names},
            },
            f"{units}_error": {
                emission: {bolo: power_error[bolo] for bolo in bolo_names},
            },
            "scaleFactor": {
                emission: {bolo: scale_factor[bolo] for bolo in bolo_names},
            },
            "toroidalRadiatedPower": {
                emission: {
                    "phi_array": list(np.linspace(0, 2 * np.pi, 20)),
                    "P_pol": [1.0] * 20,
                    "P_total": 1.0,
                }
            },
        },
    }

    path = tmp_path / "test_raddist.json"
    path.write_text(json.dumps(data))
    return str(path), bolo_names, channel_tags, emission


class TestRadDistFitting:

    def test_loads_without_error(self, tmp_path):
        path, *_ = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        assert rdf.info is not None
        assert "injectionLocation" in rdf.info

    def test_emission_names_loaded(self, tmp_path):
        path, *_ = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        assert "elongatedRing" in rdf.info["emissionNames"]

    def test_data_maps_created(self, tmp_path):
        path, bolo_names, channel_tags, emission = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        assert emission in rdf.data_maps
        for key in ("scaleFactor", "data", "data_error"):
            assert key in rdf.data_maps[emission], f"Missing key '{key}' in data_maps"

    def test_channel_keys_in_data_maps(self, tmp_path):
        path, bolo_names, channel_tags, emission = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        for ch in channel_tags:
            assert ch in rdf.data_maps[emission]["data"], f"Channel {ch} missing"

    def test_prepare_for_fits_creates_fitSynthetic(self, tmp_path):
        path, bolo_names, channel_tags, emission = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        channel_order = [channel_tags]  # one group with all channels
        rdf.prepare_for_fits(channel_order, data_max=None)
        assert hasattr(rdf, "fitSynthetic")
        assert emission in rdf.fitSynthetic

    def test_prepare_for_fits_data_length(self, tmp_path):
        n_ch = 4
        path, _, channel_tags, emission = _minimal_raddist_json(
            tmp_path, n_channels=n_ch
        )
        rdf = RadDistFitting(radDistPath=path)
        channel_order = [channel_tags]
        rdf.prepare_for_fits(channel_order)
        data_list = rdf.fitSynthetic[emission]["data"]
        assert len(data_list) == 1  # one bolometer group
        assert len(data_list[0]) == n_ch  # n_ch channels

    def test_data_max_scaling(self, tmp_path):
        """When data_max is provided, synthetic data should be scaled up."""
        path, _, channel_tags, emission = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        channel_order = [channel_tags]

        rdf.prepare_for_fits(channel_order, data_max=None)
        unscaled_max = max(max(row) for row in rdf.fitSynthetic[emission]["data"])

        rdf2 = RadDistFitting(radDistPath=path)
        rdf2.prepare_for_fits(channel_order, data_max=unscaled_max * 10.0)
        scaled_max = max(max(row) for row in rdf2.fitSynthetic[emission]["data"])

        assert scaled_max > unscaled_max

    def test_create_parameters_float_injection_location_bug(self, tmp_path):
        """
        BUG REGRESSION: When injectionLocation is a float (e.g. 120.0),
        create_parameters() builds the amplitude parameter name as 'a_120.0'.
        lmfit rejects names containing a dot, raising KeyError.

        The 'b_' parameter (line 197 of radDistFitting.py) correctly casts to
        int(), but the 'a_' parameter on line 181 does not.

        Fix: change line 181 to:
            paramName = f"a_{int(self.info['injectionLocation'])}"
        """
        path, _, channel_tags, _ = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        rdf.prepare_for_fits([channel_tags])

        # Document the bug: this currently raises KeyError because 'a_120.0'
        # is not a valid lmfit parameter name (dots are forbidden).
        with pytest.raises(KeyError, match="is not a valid Parameters name"):
            rdf.create_parameters()

    def test_create_parameters_int_injection_location_works(self, tmp_path):
        """
        create_parameters() succeeds when injectionLocation is stored as an int.
        This also verifies the b_ parameter is generated correctly.
        """
        path, _, channel_tags, _ = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        # Override to integer so the 'a_' name is valid
        rdf.info["injectionLocation"] = int(rdf.info["injectionLocation"])
        rdf.prepare_for_fits([channel_tags])
        rdf.create_parameters()

        from lmfit import Parameters

        assert isinstance(rdf.fitSynthetic["params"]["params"], Parameters)
        param_names = list(rdf.fitSynthetic["params"]["params"].keys())
        amp_params = [n for n in param_names if n.startswith("a_")]
        assert len(amp_params) >= 1
        for name in amp_params:
            assert rdf.fitSynthetic["params"]["params"][name].min >= 0.0

    def test_none_path_initialises_info_with_path_key(self):
        """RadDistFitting(None) stores {'radDistPath': None} in self.info."""
        rdf = RadDistFitting(radDistPath=None)
        assert rdf.info == {"radDistPath": None}
        assert not hasattr(rdf, "data")

    def test_missing_file_causes_keyerror_in_map_signals(self, tmp_path):
        """
        BUG: When _load_radDist() fails (file missing), self.info is not
        fully populated, but _map_signals() still runs and raises KeyError
        trying to access self.info['units'].

        Fix: _load_radDist() should set self.error_free = False or check
        hasattr/key presence in _map_signals() before proceeding.
        """
        with pytest.raises(KeyError, match="units"):
            RadDistFitting(radDistPath=str(tmp_path / "does_not_exist.json"))

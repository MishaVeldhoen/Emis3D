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
    EMIS3D_PARENT_DIRECTORY,
    EMIS3D_INPUTS_DIRECTORY,
    EMIS3D_TOKMAK_DIRECTORY,
    SUPPORTED_TOKAMAKS,
)
from main.radDistFitting import RadDistFitting

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------


class TestGlobals:

    def test_parent_directory_exists(self):
        assert EMIS3D_PARENT_DIRECTORY.is_dir(), (
            f"EMIS3D_PARENT_DIRECTORY '{EMIS3D_PARENT_DIRECTORY}' is not a directory"
        )

    def test_tokamak_directory_exists(self):
        assert EMIS3D_TOKMAK_DIRECTORY.is_dir()

    def test_supported_tokamaks_is_list(self):
        assert isinstance(SUPPORTED_TOKAMAKS, list)
        assert len(SUPPORTED_TOKAMAKS) > 0

    def test_diii_d_is_supported(self):
        assert "DIII-D" in SUPPORTED_TOKAMAKS

    def test_constants_are_path_objects(self):
        """Globals.py now uses pathlib.Path — not plain strings."""
        from pathlib import Path
        assert isinstance(EMIS3D_PARENT_DIRECTORY, Path)
        assert isinstance(EMIS3D_TOKMAK_DIRECTORY, Path)
        assert isinstance(EMIS3D_INPUTS_DIRECTORY, Path)

    def test_inputs_directory_derived_from_parent(self):
        """EMIS3D_INPUTS_DIRECTORY should be EMIS3D_PARENT_DIRECTORY / 'inputs'."""
        assert EMIS3D_INPUTS_DIRECTORY == EMIS3D_PARENT_DIRECTORY / "inputs"

    def test_tokamak_directory_derived_from_parent(self):
        assert EMIS3D_TOKMAK_DIRECTORY == EMIS3D_PARENT_DIRECTORY / "tokamaks"

    def test_all_does_not_export_os_or_path_helpers(self):
        """__all__ must only contain the four named constants."""
        import main.Globals as G
        assert hasattr(G, "__all__"), "Globals.py must define __all__"
        for name in G.__all__:
            assert not name.startswith("_"), f"__all__ exports private name: {name}"
        # os-path helpers that used to leak through the old star import
        for leaked in ("join", "dirname", "realpath", "os", "Path"):
            assert leaked not in G.__all__, (
                f"'{leaked}' should not be in __all__"
            )

    def test_emis3d_root_env_override(self, tmp_path, monkeypatch):
        """Setting EMIS3D_ROOT should redirect all path constants."""
        monkeypatch.setenv("EMIS3D_ROOT", str(tmp_path))
        # Re-import the module after env change
        import importlib, main.Globals as G
        importlib.reload(G)
        from pathlib import Path
        assert G.EMIS3D_PARENT_DIRECTORY == Path(str(tmp_path))
        assert G.EMIS3D_TOKMAK_DIRECTORY == Path(str(tmp_path)) / "tokamaks"
        assert G.EMIS3D_INPUTS_DIRECTORY == Path(str(tmp_path)) / "inputs"
        # Restore
        importlib.reload(G)


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

    def test_create_parameters_float_injection_location(self, tmp_path):
        """
        create_parameters() also works when injectionLocation is a float (120.0).
        The fix in radDistFitting.py casts it to int() so the lmfit parameter
        name is 'a_120' not the invalid 'a_120.0'.
        """
        path, _, channel_tags, _ = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        # Leave injectionLocation as 120.0 (the float stored in the fixture JSON)
        assert isinstance(rdf.info["injectionLocation"], float)
        rdf.prepare_for_fits([channel_tags])
        rdf.create_parameters()

        param_names = list(rdf.fitSynthetic["params"]["params"].keys())
        for name in param_names:
            assert "." not in name, (
                f"Parameter name '{name}' contains a dot — int() cast may be missing"
            )

    def test_none_path_initialises_info_with_path_key(self):
        """RadDistFitting(None) stores {'radDistPath': None} in self.info."""
        rdf = RadDistFitting(radDistPath=None)
        assert rdf.info == {"radDistPath": None}
        assert not hasattr(rdf, "data")

    def test_missing_file_handled_gracefully(self, tmp_path):
        """
        A missing radDist file should not crash with KeyError. The fix added a
        self._load_ok guard so _map_signals() is skipped when loading fails.
        """
        rdf = RadDistFitting(radDistPath=str(tmp_path / "does_not_exist.json"))
        assert rdf._load_ok is False
        assert not hasattr(rdf, "data")

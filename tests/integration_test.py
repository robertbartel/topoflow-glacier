"""Integration tests for Topoflow-Glacier BMI model.

This module tests the full workflow of the Topoflow-Glacier model,
including initialization, forcing data processing, model updates,
and output validation.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from pyprojroot import here

from topoflow_glacier import BmiTopoflowGlacier


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "site_prefix": "cat-3062920",
        "forcing_file": "data/sample-cat-3062920.csv",
        "dt": 1,
        "start_time": "2013032000",
        "end_time": "2013033100",
        "da": 11.418749923500716,
        "slope": 88.582729,
        "aspect": 242.8644693769529,
        "lon": -121.81418,
        "lat": 46.81953220,
        "elev": 2446.3922737596167,
        "h_active_layer": 0.125,
        "h0_snow": 5.0,
        "h0_ice": 2.0,
        "h0_swe": 0.25,
        "h0_iwe": 1.834,
        "T_rain_snow": 0.0,
    }


@pytest.fixture
def sample_forcing_data():
    """Create sample forcing data for testing."""
    return pd.read_csv(here() / "tests/data/sample-cat-3062920.csv")


@pytest.fixture
def test_environment(tmp_path, sample_config, sample_forcing_data):
    """Set up test environment with config and forcing files."""

    # Create config file
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)

    return config_file, sample_forcing_data


@pytest.fixture
def sample_outputs():
    return np.load(here() / "tests/data/output_m_total.npy").astype(np.float64)


class TestTopoflowGlacierIntegration:
    """Integration tests for the Topoflow-Glacier BMI model."""

    def test_full_model_workflow(self, test_environment, sample_outputs):
        """Test the complete model workflow from initialization to finalization."""
        config_file, forcing_df = test_environment

        model = BmiTopoflowGlacier()
        model.initialize(str(config_file))

        dest_array = np.zeros(1)
        model.get_value("snowpack__depth", dest_array)
        initial_snow = dest_array.item()
        assert initial_snow == 5.0

        model.get_value("glacier_ice__thickness", dest_array)
        initial_ice = dest_array.item()
        assert initial_ice == 2.0

        forcing_df["Time"] = pd.to_datetime(forcing_df["Time"])
        start_datetime = pd.to_datetime(model.cfg.start_time, format="%Y%m%d%H")
        end_datetime = pd.to_datetime(model.cfg.end_time, format="%Y%m%d%H")
        df = forcing_df[(forcing_df["Time"] >= start_datetime) & (forcing_df["Time"] <= end_datetime)].copy()

        print(df)
        precip_data = df["RAINRATE"].values
        temp_data = df["T2D"].values
        long_wave_radiation = df["LWDOWN"].values
        short_wave_radiation = df["SWDOWN"].values
        air_pressure = df["PSFC"].values
        air_water_vapor = df["Q2D"].values
        wind_speed = ((df["U2D"]) ** 2 + (df["V2D"]) ** 2) ** 0.5

        output_snow_melt = np.zeros(len(precip_data))
        output_ice_melt = np.zeros(len(precip_data))
        output_h_swe = np.zeros(len(precip_data))
        output_h_iwe = np.zeros(len(precip_data))
        output_h_snow = np.zeros(len(precip_data))
        output_m_total = np.zeros(len(precip_data))

        for i in range(len(precip_data)):
            model.set_value(
                "atmosphere_water__liquid_equivalent_precipitation_rate",
                np.array([precip_data[i] * 10 ** (-3)]),
            )
            model.set_value("land_surface_air__temperature", np.array([model.K_to_C + temp_data[i]]))
            model.set_value(
                "land_surface_radiation~incoming~longwave__energy_flux", np.array([long_wave_radiation[i]])
            )
            model.set_value(
                "land_surface_radiation~incoming~shortwave__energy_flux", np.array([short_wave_radiation[i]])
            )
            model.set_value("land_surface_air__pressure", np.array([air_pressure[i]]))
            model.set_value("atmosphere_air_water~vapor__relative_saturation", np.array([air_water_vapor[i]]))
            model.set_value("wind_speed_UV", np.array([wind_speed.values[i]]))

            model.update()

            dest_array = np.zeros(1)
            model.get_value("snowpack__melt_volume_flux", dest_array)
            snow_melt = dest_array.item()

            output_snow_melt[i : i + 1] = dest_array
            assert snow_melt >= 0  # Melt rate should be non-negative

            model.get_value("glacier_ice__melt_volume_flux", dest_array)
            ice_melt = dest_array.item()
            output_ice_melt[i : i + 1] = dest_array
            assert ice_melt >= 0  # Melt rate should be non-negative

            model.get_value("snowpack__depth", dest_array)
            h_snow = dest_array.item()
            output_h_snow[i : i + 1] = dest_array
            assert h_snow >= 0  # Snow depth should be non-negative

            model.get_value("glacier_ice__thickness", dest_array)
            h_ice = dest_array.item()
            assert h_ice >= 0  # Ice thickness should be non-negative

            dest_array = np.zeros(1)
            model.get_value("snowpack__liquid-equivalent_depth", dest_array)
            output_h_swe[i : i + 1] = dest_array

            dest_array = np.zeros(1)
            model.get_value("glacier__liquid_equivalent_depth", dest_array)
            output_h_iwe[i : i + 1] = dest_array

            dest_array = np.zeros(1)
            model.get_value("land_surface_water__runoff_volume_flux", dest_array)
            output_m_total[i : i + 1] = dest_array

        model.finalize()

        output_m_total = output_m_total * model.da_m2  # converting m/sec melt to m3/sec
        print(output_m_total.sum())
        assert np.allclose(
            sample_outputs,
            output_m_total,
            rtol=1e-3,
            atol=1e-6,
        ), "outputs not containing expected values"


    def test_bmi_variable_access(self, test_environment):
        """Test BMI variable getter and setter methods."""
        config_file, _ = test_environment

        model = BmiTopoflowGlacier()
        model.initialize(str(config_file))

        input_vars = model.get_input_var_names()
        assert "land_surface_air__temperature" in input_vars
        assert "atmosphere_water__liquid_equivalent_precipitation_rate" in input_vars

        output_vars = model.get_output_var_names()
        assert "snowpack__depth" in output_vars
        assert "glacier_ice__thickness" in output_vars

        var_type = model.get_var_type("snowpack__depth")
        assert "float" in var_type

        itemsize = model.get_var_itemsize("snowpack__depth")
        assert itemsize == 8  # 64-bit float

        nbytes = model.get_var_nbytes("snowpack__depth")
        assert nbytes == 8  # Single value, 8 bytes

        test_value = np.array([273.15])
        model.set_value("land_surface_air__temperature", test_value)

        retrieved = np.zeros(1)
        model.get_value("land_surface_air__temperature", retrieved)
        assert np.allclose(retrieved, test_value)

        model.finalize()


class TestTopoflowGlacierEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_snow_no_ice(self, tmp_path, sample_config):
        """Test model behavior when there's no initial snow or ice."""
        # Modify config to have no initial snow or ice
        sample_config["h0_snow"] = 0.0
        sample_config["h0_ice"] = 0.0
        sample_config["h0_swe"] = 0.0
        sample_config["h0_iwe"] = 0.0

        forcing_file = tmp_path / "test_forcing.csv"
        dates = pd.date_range(start="2013-03-20 00:00:00", periods=25, freq="h")
        forcing_df = pd.DataFrame(
            {
                "Time": dates,
                "RAINRATE": [0.0] * 25,
                "Q2D": [0.003] * 25,
                "T2D": [275.0] * 25,
                "U2D": [1.0] * 25,
                "V2D": [1.0] * 25,
                "LWDOWN": [300.0] * 25,
                "SWDOWN": [100.0] * 25,
                "PSFC": [88000.0] * 25,
            }
        )
        forcing_df.to_csv(forcing_file, index=False)
        sample_config["forcing_file"] = str(forcing_file)

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        model = BmiTopoflowGlacier()
        model.initialize(str(config_file))

        model.set_value("atmosphere_water__liquid_equivalent_precipitation_rate", np.array([0.0]))
        model.set_value("land_surface_air__temperature", np.array([5.0]))
        model.set_value("land_surface_radiation~incoming~longwave__energy_flux", np.array([300.0]))
        model.set_value("land_surface_radiation~incoming~shortwave__energy_flux", np.array([100.0]))
        model.set_value("land_surface_air__pressure", np.array([88000.0]))
        model.set_value("atmosphere_air_water~vapor__relative_saturation", np.array([0.003]))
        model.set_value("wind_speed_UV", np.array([2.0]))

        model.update()

        # Should have zero melt when there's no snow or ice
        dest_array = np.zeros(1)
        model.get_value("snowpack__melt_volume_flux", dest_array)
        snow_melt = dest_array.item()

        model.get_value("glacier_ice__melt_volume_flux", dest_array)
        ice_melt = dest_array.item()
        assert snow_melt == 0.0
        assert ice_melt == 0.0

        model.finalize()

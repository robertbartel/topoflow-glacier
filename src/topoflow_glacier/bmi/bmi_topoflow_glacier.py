from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import sys
import gc
import pickle
from numpy.typing import NDArray

from topoflow_glacier.bmi.bmi_base import BmiBase
from topoflow_glacier.bmi.config import TopoflowGlacierConfig
from topoflow_glacier.physics import solar_funcs as solar
from topoflow_glacier.physics.context import Context, build_context

from datetime import datetime, timezone
import logging
LOG = logging.getLogger("TFGLACR") # IMPORTANT! Use exact string from ewts.modules.TOPOFLOW_GLACIER_ID
try:
    from ewts.helper import getenv_any
    from ewts.logger import configure_existing_logger
    TFGLACR_USE_EWTS = True
except ImportError:
    TFGLACR_USE_EWTS = False
    
__all__ = ["BmiTopoflowGlacier"]

_dynamic_input_vars = [
    ("land_surface_radiation~incoming~longwave__energy_flux", "W m-2"),
    ("land_surface_air__pressure", "Pa"),
    ("atmosphere_air_water~vapor__relative_saturation", "kg kg-1"),
    ("atmosphere_water__liquid_equivalent_precipitation_rate", "mm h-1"),
    ("land_surface_radiation~incoming~shortwave__energy_flux", "W m-2"),
    ("land_surface_air__temperature", "degC"),
    ("land_surface_wind__x_component_of_velocity", "m s-1"),
    ("land_surface_wind__y_component_of_velocity", "m s-1"),
    # ("wind_speed_UV", "m sec-1"),

    ("ngen_realization_start_time", "s"),
    ("ngen_realization_end_time", "s"),
    ("ngen_realization_dt", "s"),
]

_calib_vars = [
    ("T_rain_snow", "degC")
]

_output_vars = [
    ("snowpack__depth", "m"),
    ("snowpack__liquid-equivalent_depth", "m"),
    ("snowpack__liquid-equivalent_mass_per_area", "kg m-2"),
    ("snowpack__melt_volume_flux", "m s-1"),
    ("glacier_ice__thickness", "m"),
    ("glacier__liquid_equivalent_depth", "m"),
    ("glacier_ice__melt_volume_flux", "m s-1"),
    ("land_surface_water__runoff_volume_flux", "m s-1"),
    ("land_surface_water__runoff_depth", "m"),
    ("atmosphere_bottom_air_water-vapor__relative_saturation", "-"),
    ("precipitation_rate", "mm s-1"),
    # NEW: discharge expected by NGen (m3 s-1)
    ("channel_water_x-section__volume_flow_rate", "m3 s-1"),

    ("atmosphere_water__snowfall_leq-volume_flux", "mm s-1"),
    ("snowpack__domain_time_integral_of_melt_volume_flux", "mm"),
    ("land_surface__temperature", "K"),
]

# --------------   Complete Name Crosswalk   -----------------------------
INTERNAL_NAME_CROSSWALK = {
    # Input variable mappings (BMI name -> internal name)
    "land_surface_radiation~incoming~longwave__energy_flux": "LW_in",
    "land_surface_air__pressure": "P_air",
    "atmosphere_air_water~vapor__relative_saturation": "Hum_sp",
    "atmosphere_water__liquid_equivalent_precipitation_rate": "P",
    "land_surface_radiation~incoming~shortwave__energy_flux": "SW_in",
    "land_surface_air__temperature": "T_air",
    "wind_speed_UV": "uz",
    # Output variable mappings (only used ones)
    "snowpack__depth": "h_snow",
    "snowpack__liquid-equivalent_depth": "h_swe",
    "snowpack__melt_volume_flux": "SM",
    "glacier_ice__thickness": "h_ice",
    "glacier__liquid_equivalent_depth": "h_iwe",
    "glacier_ice__melt_volume_flux": "IM",
    "land_surface_water__runoff_volume_flux": "M_total",
    "atmosphere_bottom_air_water-vapor__relative_saturation": "RH",
    "channel_water_x-section__volume_flow_rate": "Q_out",
    "land_surface_wind__x_component_of_velocity": "U2D",
    "land_surface_wind__y_component_of_velocity": "V2D",
    "precipitation_rate": "P_rate",

    # NEW output mappings
    "atmosphere_water__snowfall_leq-volume_flux": "P_snow",
    "snowpack__domain_time_integral_of_melt_volume_flux": "vol_SM",
    "land_surface__temperature": "T_surf",
    # Unused variables:
    # "atmosphere_bottom_air__mass-per-volume_density": "rho_air",
    # "atmosphere_bottom_air__mass-specific_isobaric_heat_capacity": "Cp_air",
    # "land_surface_net-total-energy__energy_flux": "Q_sum",
    # "water-liquid__mass-per-volume_density": "rho_H2O",
    # "snowpack__initial_domain_integral_of_liquid-equivalent_depth": "vol_swe_start",
    # "snowpack__domain_integral_of_liquid-equivalent_depth": "vol_swe",
    # "snowpack__energy-per-area_cold_content": "Eccs",
    # "snowpack__initial_depth": "h0_snow",
    # "snowpack__initial_liquid-equivalent_depth": "h0_swe",
    # "snowpack__z_mean_of_mass-per-volume_density": "rho_snow",
    # "snowpack__z_mean_of_mass-specific_isobaric_heat_capacity": "Cp_snow",
    # "glacier_ice__domain_time_integral_of_melt_volume_flux": "vol_IM",
    # "glacier__initial_domain_integral_of_liquid-equivalent_depth": "vol_iwe_start",
    # "glacier__domain_integral_of_liquid-equivalent_depth": "vol_iwe",
    # "glacier__energy-per-area_cold_content": "Ecci",
    # "glacier_ice__initial_thickness": "h0_ice",
    # "glacier__initial_liquid_equivalent_depth": "h0_iwe",
    # "glacier_ice__mass-per-volume_density": "rho_ice",
    # "glacier_ice__mass-specific_isobaric_heat_capacity": "Cp_ice",
    # "glacier_top_surface__elevation": "z_ice",
    # "cryosphere__domain_time_integral_of_melt_volume_flux": "vol_M_total",
    # # Grid and time mappings
    # "model_grid_cell__x_length": "dx",
    # "model_grid_cell__y_length": "dy",
    # "model__time_step": "dt",
}

# Reverse mapping (internal name -> BMI name)
EXTERNAL_NAME_CROSSWALK = {v: k for k, v in INTERNAL_NAME_CROSSWALK.items()}


def crosswalk_to_external(internal_name: str):
    """Return the external (BMI) name for a given internal name."""
    return EXTERNAL_NAME_CROSSWALK[internal_name]


def crosswalk_to_interal(external_name: str):
    """Return the internal name for a given external (BMI) name."""
    return INTERNAL_NAME_CROSSWALK[external_name]


def bmi_array(arr: list[float]) -> np.ndarray:
    """Trivial wrapper function to ensure the expected numpy array datatype is used."""
    return np.array(arr, dtype="float64")


def load_static_attributes(cfg: dict[str, Any], state: Context):
    for external_name in state.names():
        internal_name = crosswalk_to_interal(external_name)
        value = cfg[internal_name]
        state.set_value(external_name, bmi_array([value]))

class StdoutStyleFormatter(logging.Formatter):

    INFO_FORMAT = (
        "%(asctime)s %(name)-8s %(levelname)-7s %(message)s"
    )

    DETAILED_FORMAT = (
        "%(asctime)s %(name)-8s %(levelname)-7s "
        "%(message)s "
        "[%(filename)s.%(funcName)s(L%(lineno)s)]"
    )

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = self.INFO_FORMAT
        else:
            self._style._fmt = self.DETAILED_FORMAT

        return super().format(record)
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _configure_stdout_logging():
    LOG.setLevel(logging.INFO)

    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(StdoutStyleFormatter())
        LOG.addHandler(handler)

    LOG.propagate = False

class BmiTopoflowGlacier(BmiBase):
    """BMI composition wrapper for TopoflowGlacier"""

    def __init__(self) -> None:
        if TFGLACR_USE_EWTS:
            # Determine if running within ngen using EWTS. This must be done  
            # here when the model actually runs vs when it is imported 
            # into the ngen Python interpreter to ensure the env vars are set.
            val = getenv_any("EWTS_USE_NGEN_BRIDGE", "").strip().lower()
            if val in {"1", "true", "yes", "on"}:
                configure_existing_logger(LOG)
            else:
                _configure_stdout_logging()
                LOG.warning("ewts package installed but EWTS_USE_NGEN_BRIDGE not on. Falling back to default logging.")
        else:
            _configure_stdout_logging()

        self._dynamic_inputs = build_context(_dynamic_input_vars)
        self._calibs = build_context(_calib_vars)
        self._outputs = build_context(_output_vars)
        # Create the arrays holding serializtion data
        self._free_serialized()

        self._ngen_realization_start_time = None
        self._ngen_realization_end_time = None
        self._ngen_realization_dt = None
        self._ngen_realization_time_applied = False

    @property
    def P(self) -> np.ndarray:
        """Getter for the precipitation dynamic input state variable"""
        return self._dynamic_inputs.value("atmosphere_water__liquid_equivalent_precipitation_rate")

    @P.setter
    def P(self, value: np.ndarray) -> None:
        # BMI advertises mm h-1, but computations expect m s-1
        # initialize() defines: self.mmph_to_mps = 1/3_600_000
        self._dynamic_inputs.set_value(
            "atmosphere_water__liquid_equivalent_precipitation_rate", value * self.mmph_to_mps
        )

    @property
    def P_rate(self) -> np.ndarray:
        """Getter for the precipitation output variable in mm s-1"""
        return self._outputs.value("precipitation_rate")

    @P_rate.setter
    def P_rate(self, value: np.ndarray) -> None:
        """Setter for the precipitation output variable in mm s-1"""
        self._outputs.set_value("precipitation_rate", value)

    @property
    def T_air(self) -> np.ndarray:
        """Getter for the Air Temperature dynamic input state variable"""
        return self._dynamic_inputs.value("land_surface_air__temperature")

    @T_air.setter
    def T_air(self, value: np.ndarray) -> None:
        """Setter for the Air Temperature dynamic input state variable"""
        self._dynamic_inputs.set_value("land_surface_air__temperature", value)

    @property
    def LW_in(self) -> np.ndarray:
        """Getter for the Long Wave Radiation dynamic input state variable"""
        return self._dynamic_inputs.value("land_surface_radiation~incoming~longwave__energy_flux")

    @LW_in.setter
    def LW_in(self, value: np.ndarray) -> None:
        """Setter for the Long Wave Radiation dynamic input state variable"""
        self._dynamic_inputs.set_value("land_surface_radiation~incoming~longwave__energy_flux", value)

    @property
    def SW_in(self) -> np.ndarray:
        """Getter for the Short Wave Radiation dynamic input state variable"""
        return self._dynamic_inputs.value("land_surface_radiation~incoming~shortwave__energy_flux")

    @SW_in.setter
    def SW_in(self, value: np.ndarray) -> None:
        """Setter for the Short Wave Radiation dynamic input state variable"""
        self._dynamic_inputs.set_value("land_surface_radiation~incoming~shortwave__energy_flux", value)

    @property
    def P_air(self) -> np.ndarray:
        """Getter for the Air Pressure dynamic input state variable"""
        return self._dynamic_inputs.value("land_surface_air__pressure")

    @P_air.setter
    def P_air(self, value: np.ndarray) -> None:
        """Setter for the Air Pressure dynamic input state variable"""
        self._dynamic_inputs.set_value("land_surface_air__pressure", value)

    @property
    def Hum_sp(self) -> np.ndarray:
        """Getter for the Humidity input state variable"""
        return self._dynamic_inputs.value("atmosphere_air_water~vapor__relative_saturation")

    @Hum_sp.setter
    def Hum_sp(self, value: np.ndarray) -> None:
        """Setter for the Humidity input state variable"""
        self._dynamic_inputs.set_value("atmosphere_air_water~vapor__relative_saturation", value)

    @property
    def T_rain_snow(self) -> np.ndarray:
        """Getter for the Rain-Snow Temperature Threshold"""
        return float(self._calibs.value("T_rain_snow")[0])

    @property
    def uz(self) -> np.ndarray:
        """Wind-speed magnitude used by physics (derived or set)."""
        try:
            return self._uz
        except AttributeError:
            self._recompute_wind_speed()
            return self._uz
    @uz.setter
    def uz(self, value):
        if isinstance(value, (int, float)):
            self._uz = np.array([value], dtype=np.float64)
        else:
            self._uz = np.array(value, copy=True, dtype=np.float64)

    @property
    def wind_u(self) -> np.ndarray:
        """Wind-speed magnitude in the X direction."""
        return self._dynamic_inputs.value("land_surface_wind__x_component_of_velocity")
    @wind_u.setter
    def wind_u(self, value):
        if isinstance(value, (int, float)):
            value = np.array([value], dtype=np.float64)
        self._dynamic_inputs.set_value("land_surface_wind__x_component_of_velocity", value)

    @property
    def wind_v(self) -> np.ndarray:
        """Wind-speed magnitude in the Y direction."""
        return self._dynamic_inputs.value("land_surface_wind__y_component_of_velocity")
    @wind_v.setter
    def wind_v(self, value):
        if isinstance(value, (int, float)):
            value = np.array([value], dtype=np.float64)
        self._dynamic_inputs.set_value("land_surface_wind__y_component_of_velocity", value)

    @property
    def runoff_depth(self) -> np.ndarray:
        """Getter for the runoff depth (m) variable"""
        return self._outputs.value("land_surface_water__runoff_depth")

    @runoff_depth.setter
    def runoff_depth(self, value: np.ndarray) -> None:
        """Setter for runoff depth (m)."""
        self._outputs.set_value("land_surface_water__runoff_depth", value)


    @property
    def SM(self) -> np.ndarray:
        """Getter for the Snow Melt state variable"""
        return self._outputs.value("snowpack__melt_volume_flux")

    @SM.setter
    def SM(self, value: np.ndarray) -> None:
        """Setter for the Snow Melt state variable"""
        self._outputs.set_value("snowpack__melt_volume_flux", value)

    @property
    def IM(self) -> np.ndarray:
        """Getter for the Ice Melt state variable"""
        return self._outputs.value("glacier_ice__melt_volume_flux")

    @IM.setter
    def IM(self, value: np.ndarray) -> None:
        """Setter for the Ice Melt state variable"""
        self._outputs.set_value("glacier_ice__melt_volume_flux", value)

    @property
    def h_swe(self) -> np.ndarray:
        """Getter for the Snow Water Equivalent Height state variable"""
        return self._outputs.value("snowpack__liquid-equivalent_depth")

    @h_swe.setter
    def h_swe(self, value: np.ndarray) -> None:
        """Setter for the Snow Water Equivalent Height state variable"""
        self._outputs.set_value("snowpack__liquid-equivalent_depth", value)

    @property
    def h_iwe(self) -> np.ndarray:
        """Getter for the Ice Water Equivalent Height state variable"""
        return self._outputs.value("glacier__liquid_equivalent_depth")

    @h_iwe.setter
    def h_iwe(self, value: np.ndarray) -> None:
        """Setter for the Ice Water Equivalent Height state variable"""
        self._outputs.set_value("glacier__liquid_equivalent_depth", value)

    @property
    def h_snow(self) -> np.ndarray:
        """Getter for the Snow Height state variable"""
        return self._outputs.value("snowpack__depth")

    @h_snow.setter
    def h_snow(self, value: np.ndarray) -> None:
        """Setter for the Snow Height state variable"""
        self._outputs.set_value("snowpack__depth", value)

    @property
    def h_ice(self) -> np.ndarray:
        """Getter for the Ice Height state variable"""
        return self._outputs.value("glacier_ice__thickness")

    @h_ice.setter
    def h_ice(self, value: np.ndarray) -> None:
        """Setter for the Ice Height state variable"""
        self._outputs.set_value("glacier_ice__thickness", value)

    @property
    def M_total(self) -> np.ndarray:
        """Getter for the melt (runoff) state variable"""
        return self._outputs.value("land_surface_water__runoff_volume_flux")

    @M_total.setter
    def M_total(self, value: np.ndarray) -> None:
        """Setter for the melt (runoff) state variable"""
        self._outputs.set_value("land_surface_water__runoff_volume_flux", value)

    @property
    def RH(self) -> np.ndarray:
        """Getter for the relative humidity state variable"""
        return self._outputs.value("atmosphere_bottom_air_water-vapor__relative_saturation")

    @RH.setter
    def RH(self, value: np.ndarray) -> None:
        """Setter for the relative humidity state variable"""
        self._outputs.set_value("atmosphere_bottom_air_water-vapor__relative_saturation", value)

    def _sync_internal_outputs(self) -> None:
        """Copy internal model variables into the BMI output context."""

        self._outputs.set_value(
            "snowpack__liquid-equivalent_mass_per_area",
            np.asarray(self.h_swe * self.rho_H2O, dtype="float64").reshape(-1),
        )

        self._outputs.set_value(
            "atmosphere_water__snowfall_leq-volume_flux",
            np.asarray(self.P_snow * 1000.0, dtype="float64").reshape(-1),   # m/s -> mm/s
        )

        snow_melt_mm = (self.vol_SM / self.da_m2) * 1000.0    # m -> mm, if vol_SM is m3 over area
        self._outputs.set_value(
            "snowpack__domain_time_integral_of_melt_volume_flux",
            np.asarray(snow_melt_mm, dtype="float64").reshape(-1),
        )
        self._outputs.set_value(
            "land_surface__temperature",
            np.asarray(self.T_surf + 273.15, dtype="float64").reshape(-1),   # degC -> K
        )

    def initialize(self, config_file: str | Path) -> None:
        """Initialize the BMI model and pre-compute all bookkeeping needed by the adapter."""
        LOG.info("initialize")

        # --- load config (YAML -> TopoflowGlacierConfig) ---
        with open(config_file) as f:
            cfg_dict = yaml.safe_load(f)

        LOG.info(f"bmi config file : {config_file}")

        for key in ("start_time", "end_time"):
            if key in cfg_dict and cfg_dict[key] is not None and not isinstance(cfg_dict[key], str):
                cfg_dict[key] = str(cfg_dict[key])

        self.cfg: TopoflowGlacierConfig = TopoflowGlacierConfig.model_validate(cfg_dict)

        # --- constants & unit helpers ---
        self.hours_per_day = np.float64(24)
        self.seconds_per_Day = np.float64(86400)
        self.sec_per_year = np.float64(31536000)
        self.mps_to_mmph = np.float64(3600000)
        self.mmph_to_mps = np.float64(1.0) / 3600000.0
        self.C_to_K = 273.15
        self.K_to_C = -273.15
        self.twopi = np.float64(2) * np.pi
        self.one_seventh = np.float64(1) / 7

        # --- spatial constants ---
        self.da_km2 = np.float64(self.cfg.da)
        self.da_m2 = self.da_km2 * 1.0e6
        self.slopes = (
            np.array([self.cfg.slope], dtype="float64")
            if np.isscalar(self.cfg.slope)
            else np.asarray(self.cfg.slope, dtype="float64")
        )

        # --- timestep normalization: ensure dt is seconds ---
        self.dt = float(self.cfg.dt)
        if self.dt <= 10.0:
            LOG.warning(f"dt={self.dt} looks like HOURS; converting to seconds (dt *= 3600).")
            self.dt *= 3600.0
        self.days_per_dt = self.dt / 86400.0
        self._timestep_size_s = float(self.dt)

        # --- initialize adapter time state ---
        self._adapter_time_configured = False
        self._adapter_start_time_s = 0.0
        self._run_end_time_s = None
        self._adapter_end_time_s = None

        if not self._adapter_time_configured:
            if self.cfg.start_time is not None and self.cfg.end_time is not None:
                start_dt = self._parse_iso_like(self.cfg.start_time)
                end_dt = self._parse_iso_like(self.cfg.end_time)

                self._recompute_adapter_time_bounds(start_dt, end_dt)

                LOG.info(
                    "Using fallback config time: start=%s end=%s dt=%gs",
                    self.start_datetime, self.end_datetime, self._timestep_size_s
                )
            else:
                start_dt = pd.Timestamp("1970-01-01 00:00:00")
                end_dt = start_dt + pd.Timedelta(seconds=float(self._timestep_size_s))

                self._recompute_adapter_time_bounds(start_dt, end_dt)

                LOG.info(
                    "Using placeholder initialization time until ngen provides realization time: "
                    "start=%s end=%s dt=%gs",
                    self.start_datetime, self.end_datetime, self._timestep_size_s
                )

        # --- dynamic input & output contexts ---
        self.T_surf = np.array([0.0], dtype="float64")
        self.RH = np.array([0.0], dtype="float64")
        self.p0 = np.array([0.0], dtype="float64")
        self.z = np.array([10.0], dtype="float64")
        self.cloud_factor = np.array([0.0], dtype="float64")
        self.canopy_factor = np.array([0.0], dtype="float64")
        self.P_rain = np.array([0.0], dtype="float64")
        self.P_snow = np.array([0.0], dtype="float64")
        self.e_air = np.array([1e-6], dtype="float64")
        self.e_surf = np.array([1e-6], dtype="float64")
        self.em_air = np.array([0.0], dtype="float64")
        self.Qn_SW = np.array([0.0], dtype="float64")
        self.Qn_LW = np.array([0.0], dtype="float64")
        self.Q_sum = np.array([0.0], dtype="float64")
        self.Qc = np.array([0.0], dtype="float64")
        self.Qa = np.array([0.0], dtype="float64")
        self.Qe = np.array([0.0], dtype="float64")
        self.P_max = np.array([0.0], dtype="float64")
        self.vol_P = np.array([0.0], dtype="float64")
        self.vol_PR = np.array([0.0], dtype="float64")
        self.vol_PS = np.array([0.0], dtype="float64")
        self.Qn_tot = np.array([0.0], dtype="float64")

        # --- ice/snow constants ---
        self.rho_H2O = np.float64(self.cfg.rho_H2O)
        self.rho_ice = np.float64(self.cfg.rho_ice)
        self.Cp_ice = np.float64(self.cfg.Cp_ice)
        self.g = np.float64(self.cfg.g)
        self.Qg = np.float64(self.cfg.geothermal_heat_flux)
        self.grad_Tz = np.float64(self.cfg.geothermal_gradient)

        self.rho_snow = np.float64(self.cfg.rho_snow)
        self.Cp_snow = np.float64(self.cfg.Cp_snow)
        self.Lf = np.float64(self.cfg.Lf)
        self._calibs.set_value("T_rain_snow", np.array([self.cfg.T_rain_snow], dtype="float64"))

        # --- state variables ---
        self.T0 = np.array([self.cfg.T0], dtype="float64")
        self.h_active_layer = np.array([self.cfg.h_active_layer], dtype="float64")
        self.mr_ice = np.array([0.0], dtype="float64")
        self.vol_MR = np.array([0.0], dtype="float64")
        self.meltrate = np.array([0.0], dtype="float64")

        self._outputs.set_value("snowpack__depth", np.array([self.cfg.h0_snow], dtype="float64"))
        self._outputs.set_value("glacier_ice__thickness", np.array([self.cfg.h0_ice], dtype="float64"))
        self._outputs.set_value("snowpack__liquid-equivalent_depth", np.array([self.cfg.h0_swe], dtype="float64"))
        self._outputs.set_value(
            "snowpack__liquid-equivalent_mass_per_area",
            np.array([self.cfg.h0_swe * self.rho_H2O], dtype="float64"),
        )
        self._outputs.set_value("glacier__liquid_equivalent_depth", np.array([self.cfg.h0_iwe], dtype="float64"))
        self._outputs.set_value("channel_water_x-section__volume_flow_rate", np.array([0.0], dtype="float64"))
        self._outputs.set_value("precipitation_rate", np.array([0.0], dtype="float64"))

        # melt-volume accumulators
        self.vol_SM = np.array([0.0], dtype="float64")
        self.vol_IM = np.array([0.0], dtype="float64")
        self.vol_M_total = np.array([0.0], dtype="float64")
        self.vol_swe = np.array([0.0], dtype="float64")
        self.vol_swe_start = np.array([0.0], dtype="float64")
        self.vol_iwe = np.array([0.0], dtype="float64")
        self.vol_iwe_start = np.array([0.0], dtype="float64")

        # albedo & snowfall buffer
        self.albedo = np.array([0.3], dtype="float64")
        self._init_three_day_snow_buffer()
        self.n = np.array([0.0], dtype="float64")

        # density ratios
        self.ws_density_ratio = self.rho_H2O / self.rho_snow
        self.wi_density_ratio = self.rho_H2O / self.rho_ice

        # cold content
        self.T0_cc = self.T0
        T_snow = self.T_surf
        del_T = self.T0_cc - T_snow
        self.Eccs = (self.rho_snow * self.Cp_snow) * self.h_snow * del_T
        self.Eccs = np.maximum(self.Eccs, np.array([0.0]))
        self.Ecci = (self.rho_ice * self.Cp_ice) * self.h_active_layer * del_T
        self.Ecci = np.maximum(self.Ecci, np.array([0.0]))

        self._finalized = False

        # julian day seed
        self.year = self.start_datetime.year
        self.julian_day = solar.Julian_Day(
            self.start_datetime.month,
            self.start_datetime.day,
            self.start_datetime.hour,
            year=self.year
        )

        # --- previous storages ---
        self.previous_swe = np.array(self.h_swe, dtype="float64").copy()
        self.previous_iwe = np.array(self.h_iwe, dtype="float64").copy()

        # --- slope & aspect ---
        self.set_aspect_angle()
        self.set_slope_angle()

        # --- initial volumes ---
        self.vol_swe[:] = np.sum(np.float64(self.h_swe * self.cfg.da))
        self.vol_iwe[:] = np.sum(np.float64(self.h_iwe * self.cfg.da))

        # --- time-step index ---
        self._timestep = 0
        self._t_index = 0

        self._skip_solar_geometry = True

        self._sync_internal_outputs()
        LOG.debug(f"Output vars : {self.get_output_var_names()}")

        LOG.info("initialize complete")

    def _init_three_day_snow_buffer(self) -> None:
        """
        Initialize the rolling 3-day buffer (in *timesteps*) used by the albedo
        routine to track recent snowfall. Works for any dt (seconds).
        """
        secs_3days = 3 * 24 * 3600
        n_steps_3days = max(1, int(np.ceil(secs_3days / float(self.dt))))
        self.P_snow_3day_watershed = np.zeros(n_steps_3days, dtype="float64")

    def update(self) -> None:
        """Advance exactly one dt without exceeding run end; safe for adapter fencepost."""
        LOG.debug("update")

        if not self._adapter_time_configured:
            raise RuntimeError("TopoFlow-Glacier: realization time not set before update")

        dt = float(self.get_time_step())
        t_now = self.get_current_time()

        # If we're already at/after true run end, no-op but snap index to the end.
        run_end = float(getattr(self, "_run_end_time_s", 0.0))
        if t_now > (run_end + 1e-12):
            LOG.info("Reached run end (forcing exhausted); no-op update.")
            self._timestep = int(getattr(self, "_n_steps", 0))
            if hasattr(self, "_t_index"):
                self._t_index = int(getattr(self, "_n_steps", 0))
            return

        # -------------------------
        # Meteorology / Energy part
        # -------------------------
        self.update_atm_pressure_from_elevation(T_C=True, MBAR=True)
        self.update_P_integral()
        self.update_P_max()
        self.update_P_rain()
        self.update_P_snow()
        self.update_P_rain_integral()
        self.update_P_snow_integral()
        self.update_saturation_vapor_pressure(MBAR=True)
        self.update_vapor_pressure_from_spHum_AirPre(MBAR=True)
        self.update_RH()
        self.update_dew_point()
        self.update_T_surf()
        self.update_saturation_vapor_pressure(MBAR=True, SURFACE=True)
        self.update_bulk_richardson_number()
        self.update_bulk_aero_conductance()
        self.update_sensible_heat_flux()
        self.update_precipitable_water_content()
        self.update_vapor_pressure(SURFACE=True)
        self.update_latent_heat_flux()
        self.update_conduction_heat_flux()
        self.update_advection_heat_flux()
        self.update_julian_day()  # Run using seconds
        self.update_albedo(method="aging")
        self.set_aspect_angle()
        self.set_slope_angle()
        self.update_net_shortwave_radiation()
        self.update_em_air()
        self.update_net_longwave_radiation()
        self.update_net_energy_flux()

        # -------------------------
        # Snow & Ice melt
        # -------------------------
        self.extract_previous_swe()
        self.extract_previous_snow_depth()
        self.update_snow_meltrate()  # (meltrate = SM)
        self.enforce_max_snow_meltrate()  # (before SM integral!)
        self.update_SM_integral()
        self.update_swe()
        self.update_snowfall_cold_content()
        self.update_ice_meltrate()
        self.enforce_max_ice_meltrate()
        self.update_IM_integral()
        self.update_iwe()  # relies on previous timestep's swe value
        self.update_combined_meltrate()
        self.update_ws_density_ratio()
        self.update_snow_depth()  
        self.update_wi_density_ratio()
        self.update_ice_depth()
        self.update_snowpack_cold_content()
        self._sync_internal_outputs()

        # advance index AFTER computing step diagnostics
        self._timestep += 1
        self._t_index += 1

        # best-effort debug line for one-cell runs
        try:
            LOG.debug(
                "Qsum=%.3f W/m2, SM=%.6e m/s, IM=%.6e m/s, P_rain=%.6e m/s",
                float(np.asarray(self.Q_sum).reshape(-1)[0]),
                float(np.asarray(self.SM).reshape(-1)[0]),
                float(np.asarray(self.IM).reshape(-1)[0]),
                float(np.asarray(self.P_rain).reshape(-1)[0]),
            )
        except Exception:
            pass

    def finalize(self) -> None:
        """
        Clean up any internal resources of the model.

        - Idempotent (safe to call multiple times).
        - Avoids heavy work when the Python interpreter is shutting down.
        - Drops large arrays/contexts to help GC and reduce teardown issues.
        """
        # If we've already finalized this instance, do nothing.
        if getattr(self, "_finalized", False):
            LOG.debug("finalize: already finalized; skipping.")
            return

        # Mark as finalized **first** so even if something below goes wrong
        # we won't re-enter from a second call.
        self._finalized = True

        # Best-effort: if the interpreter is in the middle of shutting down,
        # avoid touching anything complicated (logging, numpy, etc.).
        try:
            is_finalizing = getattr(sys, "is_finalizing", None)
            if callable(is_finalizing) and is_finalizing():
                # Don't do any heavy cleanup; the interpreter is already
                # tearing everything down.
                return
        except Exception:
            # If anything goes wrong here, just continue with a minimal cleanup.
            pass

        LOG.info("finalize: starting cleanup of Topoflow-Glacier BMI instance.")

        # Best-effort cleanup — all inside a big try so we never raise.
        try:
            # Drop references to big arrays / state that NGen will no longer use.
            # This mainly helps with memory and keeps GC simple.
            attrs_to_clear = [
                "_dynamic_inputs",
                "_calibs",
                "_outputs",
                "_serialized",
                "cfg",
                "slopes",
                "P_snow_3day_watershed",
                "T_rain_snow", "T_surf", "RH", "p0", "z",
                "cloud_factor", "canopy_factor",
                "P_rain", "P_snow",
                "e_air", "e_surf",
                "em_air",
                "Qn_SW", "Qn_LW",
                "Q_sum", "Qc", "Qa", "Qe", "Qh",
                "P_max", "vol_P", "vol_PR", "vol_PS",
                "Qn_tot",
                "T0", "h_active_layer",
                "mr_ice", "vol_MR", "meltrate",
                "vol_SM", "vol_IM", "vol_M_total",
                "vol_swe", "vol_swe_start",
                "vol_iwe", "vol_iwe_start",
                "albedo", "n",
                "Eccs", "Ecci",
                "h_snow", "h_swe",
                "h_ice", "h_iwe",
                "M_total", "SM", "IM",
                "previous_swe", "previous_iwe",
                "start_datetime", "end_datetime",
            ]

            for name in attrs_to_clear:
                if hasattr(self, name):
                    try:
                        setattr(self, name, None)
                    except Exception:
                        # Don't let any single attribute break finalize
                        pass

            # Optional: encourage garbage collection once we've dropped references.
            try:
                gc.collect()
            except Exception:
                pass

            LOG.info("finalize: cleanup complete.")
        except Exception as e:
            # Never propagate exceptions out of finalize; just log if we still can.
            try:
                LOG.warning(f"finalize: swallowed exception during cleanup: {e!r}")
            except Exception:
                # Logging itself might fail late in teardown; ignore.
                pass


    def update_until(self, until: float) -> None:
        LOG.debug("update_until")
        dt = self.get_time_step()
        end = self.get_end_time()
        target = min(float(until), float(end))
        t = self.get_current_time()
        if t >= target:
            LOG.info("target reached")
            return
        remaining = max(0.0, target - t)
        n_steps = int(np.floor((remaining + 1e-12) / dt))
        for _ in range(n_steps):
            self.update()
    
    def _parse_iso_like(self, s: str) -> datetime:
        """
        Parse a realization-provided datetime string.
        Accepts formats like:
          'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD HH',
          'YYYY-MM-DD', or compact 'YYYYMMDDHH'.
        Raises ValueError if unrecognized.
        """
        s = str(s).strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d %H",
                    "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass
        try:
            # compact fallback
            return datetime.strptime(s.replace("-", ""), "%Y%m%d%H")
        except ValueError:
            raise ValueError(f"Unrecognized datetime string: {s!r}")

    def _recompute_adapter_time_bounds(self, start_dt: datetime, end_dt: datetime) -> None:
        """
        Given concrete datetimes and the already-known dt (seconds), recompute
        adapter/bookkeeping times so get_start_time/get_end_time are consistent.
        """
        dt = float(self._timestep_size_s)
        if dt <= 0.0:
            raise ValueError("Time step (dt) must be positive before setting time bounds.")

        if end_dt <= start_dt:
            raise ValueError("End time must be strictly after start time.")

        total_seconds = float((end_dt - start_dt).total_seconds())
        n_full = int(np.floor(total_seconds / dt + 1e-12))

        self._n_steps = n_full + 1
        self._adapter_start_time_s = 0.0
        self._run_end_time_s = float(n_full) * dt
        self._adapter_end_time_s = float(n_full + 1) * dt

        self.start_datetime = pd.to_datetime(start_dt)
        self.end_datetime   = pd.to_datetime(end_dt)
        self.start_year, self.start_month, self.start_day, self.start_hour = (
            self.start_datetime.year, self.start_datetime.month, self.start_datetime.day, self.start_datetime.hour
        )
        self.end_year, self.end_month, self.end_day, self.end_hour = (
            self.end_datetime.year, self.end_datetime.month, self.end_datetime.day, self.end_datetime.hour
        )

        self._adapter_time_configured = True

        LOG.info(
            "Realization time applied: start=%s end=%s dt=%gs n_steps=%d "
            "(run_end=%gs, adapter_end=%gs)",
            self.start_datetime, self.end_datetime, dt, self._n_steps,
            self._run_end_time_s, self._adapter_end_time_s
        )

    def _datetime_from_epoch_seconds(self, value: float) -> datetime:
        return pd.to_datetime(float(value), unit="s", utc=True).tz_convert(None).to_pydatetime()

    def _try_apply_ngen_realization_time(self) -> None:
        if self._ngen_realization_time_applied:
            return

        if (
            self._ngen_realization_start_time is None
            or self._ngen_realization_end_time is None
            or self._ngen_realization_dt is None
        ):
            return

        start_epoch = float(self._ngen_realization_start_time)
        end_epoch = float(self._ngen_realization_end_time)
        dt_seconds = float(self._ngen_realization_dt)

        if start_epoch <= 0.0 or end_epoch <= 0.0 or dt_seconds <= 0.0:
            return

        self._timestep_size_s = dt_seconds
        self.dt = dt_seconds
        self.days_per_dt = self.dt / 86400.0

        start_dt = self._datetime_from_epoch_seconds(start_epoch)
        end_dt = self._datetime_from_epoch_seconds(end_epoch)

        self._recompute_adapter_time_bounds(start_dt, end_dt)

        self._timestep = 0
        self._t_index = 0

        self.year = self.start_datetime.year
        self.julian_day = solar.Julian_Day(
            self.start_datetime.month,
            self.start_datetime.day,
            self.start_datetime.hour,
            year=self.year
        )

        self._ngen_realization_time_applied = True

        LOG.info(
            "TopoFlow-Glacier realization time applied from ngen BMI inputs: "
            "start=%s end=%s dt=%gs",
            self.start_datetime,
            self.end_datetime,
            self._timestep_size_s
        )

    def get_start_time(self) -> float:
        """BMI: start time in seconds since model epoch (0 for this run)."""
        start_s = getattr(self, "_adapter_start_time_s", None)
        if start_s is None:
            start_s = 0.0
        LOG.debug("get_start_time: %s", start_s)
        return float(start_s)

    def get_end_time(self) -> float:
        """
        BMI: end time in seconds. We prefer adapter fencepost if present,
        else last valid run time, else fall back to _n_steps * dt.
        """
        end_s = getattr(self, "_adapter_end_time_s", None)
        if end_s is None:
            end_s = getattr(self, "_run_end_time_s", None)
        if end_s is None:
            nsteps = int(getattr(self, "_n_steps", 0))
            end_s = nsteps * self.get_time_step()
        LOG.debug("get_end_time: %s", end_s)
        return float(end_s)

    def get_time_step(self) -> float:
        LOG.debug(f"get_time_step: {self._timestep_size_s}")
        #return float(self._timestep_size_s)
        dt = float(getattr(self, "_timestep_size_s", getattr(self, "dt", 0.0)))
        if dt <= 0.0:
            # Fallback so adapter never sees 0
            dt = 3600.0
        LOG.debug("get_time_step: %s", dt)
        return dt

    def get_time_units(self) -> str:
        LOG.debug("get_time_units")
        return "s"

    def get_current_time(self) -> float:
        """Current model time in seconds since start, based on internal step index."""
        dt = float(self.get_time_step())
        t = float(self._t_index) * dt
        LOG.debug(f"get_current_time: t_index={self._t_index}, t={t}")
        return t

    def is_at_end_time(self) -> bool:
        LOG.info(f"is_at_end_time  {self.get_end_time()}")

        return self.get_current_time() >= (self.get_end_time() - 1e-12)

    def _parse_yyyymmddhh(self, s: str) -> tuple[int, int, int, int]:
        """Parse a variety of 'start_time'/'end_time' strings into (year, month, day, hour).
        Accepted formats:
          - 'YYYY-MM-DD HH:MM:SS'
          - 'YYYY-MM-DD HH:MM'
          - 'YYYY-MM-DDTHH:MM:SS'
          - 'YYYYMMDDHH'
          - 'YYYYMMDD-HH'
        """
        from datetime import datetime

        s = str(s).strip()

        # Compact numeric forms
        if len(s) == 10 and s.isdigit():
            # 'YYYYMMDDHH'
            return int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10])

        if len(s) == 11 and s[8] == '-' and s.replace('-', '').isdigit():
            # 'YYYYMMDD-HH'
            return int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[9:11])

        # Flexible strptime attempts
        for fmt in ("%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.year, dt.month, dt.day, dt.hour
            except ValueError:
                pass

        # If we get here, we don't recognize the format
        LOG.critical(f"Unrecognized datetime format: {s!r}")
        raise ValueError(f"Unrecognized datetime format: {s!r}")

    def _recompute_wind_speed(self) -> None:
        self.uz = (self.wind_u ** 2 + self.wind_v ** 2) ** 0.5

    def update_atm_pressure_from_elevation(self, T_C=True, MBAR=False):
        """
        Estimate atmospheric pressure at elevation z_m (meters).

        Parameters
        ----------
        z_m : float
            Elevation above sea level [m].
        T_C : float, optional
            Air temperature [°C] for the isothermal exponential model.
            If None, uses the standard atmosphere formula with lapse rate.
        MBAR : bool, optional
            If True, return pressure in hPa (mbar). Default False = Pa.

        Returns
        -------
        p0 : float
            Atmospheric pressure [Pa]
        """
        LOG.debug("update_atm_pressure_from_elevation")
        # constants
        sea_level_p0 = self.cfg.sea_level_p0  # sea-level standard pressure [Pa]
        T0 = self.cfg.sea_level_T0  # sea-level standard temperature [K]
        g = self.cfg.g  # gravity [m/s2]
        L = self.cfg.T_lapse_rate  # temperature lapse rate [K/m]
        R_star = self.cfg.uni_gas_const  # universal gas constant [J/mol/K]
        M = self.cfg.M_mass_air  # molar mass of dry air [kg/mol]

        if not T_C:
            # Standard atmosphere with lapse rate
            self.p0 = sea_level_p0 * (1 - (L * self.cfg.elev) / T0) ** (g * M / (R_star * L))  # Pa
        else:
            # Isothermal assumption with given T in Celsius
            T_K = self.T_air + 273.15
            self.p0 = sea_level_p0 * np.exp(-M * g * self.cfg.elev / (R_star * T_K))  # Pa
        self.p0 = self.p0 / np.float64(1000)  # [kPa]

        if MBAR:  # KPa to mbar
            self.p0 = self.p0 * np.float64(10.0)

    def update_P_integral(self):
        """Update mass total for P, sum over all pixels
        -------------------------------------------------
        We need to include total precip here, that is,
        P = P_rain + P_snow (liquid equivalent), not
        just P_rain, for use in a mass balance check.
        P_rain and da are both either scalar or grid.
        -------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_p_integral")
        volume = np.double(self.P * self.da_m2 * self.dt)  # [m^3 in the unit of self.dt]
        self.vol_P += np.sum(volume)
        P_output_mms = self.P * 1000.0  # [m s-1] -> [mm s-1] 
        self._outputs.set_value("precipitation_rate", P_output_mms)

    def update_P_max(self):
        """Save the maximum precip. rate in [m/s]
        -------------------------------------------
        Must use "fill()" to preserve reference.
        -------------------------------------------
        """  # noqa: D205
        LOG.debug("update_p_max")
        self.P_max.fill(np.maximum(self.P_max, self.P.max()))

    def update_P_rain(self):
        """P_rain is the precip that falls as liquid that
        can contribute to runoff production.
        -------------------------------------------------
        P_rain is used by channel_base.update_R.
        -------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_p_rain")
        P_rain = self.P * (self.T_air > self.T_rain_snow)

        if np.ndim(self.P_rain) == 0:
            self.P_rain.fill(P_rain)  #### (mutable scalar)
        else:
            self.P_rain = P_rain

    def update_P_snow(self):
        """P_snow is the precip that falls as snow or ice
        that contributes to the snow depth.  This snow
        may melt to contribute to runoff later on.
        -------------------------------------------------
        P_snow is a "water equivalent" volume flux
        that was determined from a total volume flux
        and a rain-snow temperature threshold.
        -------------------------------------------------
        P_snow is used by snow_base.update_depth.
        -------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_p_snow")
        self.P_snow = self.P * (self.T_air <= self.T_rain_snow)

    def update_P_rain_integral(self):
        """Update mass total for P, sum over all pixels
        ------------------------------------------------
        2023-08-31. This one only uses P_rain.
        P_rain and da are both either scalar or grid.
        ------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_p_rain_integeral")
        volume = np.double(self.P_rain * self.da_m2 * self.dt)  # [m^3]
        self.vol_PR += np.sum(volume)

    def update_P_snow_integral(self):
        """Update mass total for P_snow, sum over all pixels
        # ----------------------------------------------------
        # 2023-09-11. This one only uses P_snow.
        # P_snow and da are both either scalar or grid.
        # ------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_p_snow_integeral")
        volume = np.double(self.P_snow * self.da_m2 * self.dt)  # [m^3]
        self.vol_PS += np.sum(volume)

    def update_bulk_richardson_number(self):
        """
        (9/6/14)  Found a typo in the Zhang et al. (2000) paper,
        in the definition of Ri.  Also see Price and Dunne (1976).
        We should have (Ri > 0) and (T_surf > T_air) when STABLE.
        This also removes problems/singularities in the corrections
        for the stable and unstable cases in the next function.
        ---------------------------------------------------------------
        Notes: Other definitions are possible, such as the one given
               by Dingman (2002, p. 599).  However, this one is the
               one given by Zhang et al. (2000) and is meant for use
               with the stability criterion also given there.
        ---------------------------------------------------------------
        """
        LOG.debug("update_bulk_richardson_number")
        top = self.g * self.z * (self.T_air - self.T_surf)
        bot = (self.uz) ** 2.0 * (self.T_air + np.float64(273.15))
        bot = np.asarray(bot, dtype="float64")
        # prevent division by zero
        bot = np.where(bot == 0.0, 0.01, bot)
        self.Ri = top / bot

    def update_bulk_aero_conductance(self):
        """Notes: Dn       = bulk exchange coeff for the conditions of
                          neutral atmospheric stability [m/s]
               Dh       = bulk exchange coeff for heat  [m/s]
               De       = bulk exchange coeff for vapor [m/s]
               h_snow   = snow depth [m]
               z0_air   = surface roughness length scale [m]
                          (includes vegetation not covered by snow)
               z        = height that has wind speed uz [m]
               uz       = wind speed at height z [m/s]
               kappa    = 0.408 = von Karman's constant [unitless]
               RI       = Richardson's number (see function)
        ----------------------------------------------------------------
        Compute bulk exchange coeffs (neutral stability)
        using the logarithm "law of the wall".
        -----------------------------------------------------
        Note that "arg" = the drag coefficient (unitless).
        -----------------------------------------------------
        Dn will be a grid if any of the variables:
          z, h_snow, z0_air, or uz is a grid.
        -----------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_bulk_aero_conductance")
        h_snow = self.h_snow  # (ref from new framework)

        # Guard against non-positive argument to log
        denom = np.maximum((self.z - h_snow) / self.cfg.z0_air, 0.01)
        arg = self.cfg.kappa / np.log(denom)
        Dn = self.uz * (arg) ** 2.0

        # Treat “neutral” if air–surface delta-T is (nearly) zero everywhere
        # (elementwise compare with tolerance to avoid array truth ambiguities)
        delta_T = np.asarray(self.T_air) - np.asarray(self.T_surf)
        neutral_everywhere = np.all(np.isfinite(delta_T)) and np.all(np.abs(delta_T) < 1e-12)

        if neutral_everywhere:
            # --------------------------------------------
            # All pixels are neutral. Set Dh = De = Dn.
            # --------------------------------------------
            self.Dn = Dn
            self.Dh = Dn
            self.De = Dn
            return

        Dh = np.array(Dn, copy=True)  ### (9/7/14.  Save Dn also.)
        Ri = np.asarray(self.Ri)
        nD = Dh.size
        nR = Ri.size
        if nR > 1:
            # --------------------------
            # Case where RI is a grid
            # --------------------------
            ws = Ri > 0  # where stable
            ns = int(np.sum(ws))
            wu = np.logical_not(ws)  # where unstable
            nu = int(np.sum(wu))

            if nD == 1:
                # -----------------------------------
                # Convert Dh to a grid, same as Ri
                # -----------------------------------
                Dh = Dh + np.zeros(Ri.shape, dtype="float64")

            # ----------------------------------------------------------
            # If (Ri > 0), or (T_surf > T_air), then STABLE. (9/6/14)
            # ----------------------------------------------------------
            # When ws and wu are boolean arrays, don't
            # need to check whether any are True.
            # -------------------------------------------
            # Dh[ws] = Dh[ws] / (np.float64(1) + (np.float64(10) * self.Ri[ws]))
            # Dh[wu] = Dh[wu] * (np.float64(1) - (np.float64(10) * self.Ri[wu]))
            # -----------------------------------------------------------------------
            if ns != 0:
                Dh[ws] = Dh[ws] / (np.float64(1) + (np.float64(10) * Ri[ws]))
            if nu != 0:
                Dh[wu] = Dh[wu] * (np.float64(1) - (np.float64(10) * Ri[wu]))
        else:
            # ----------------------------
            # Case where Ri is a scalar
            # --------------------------------
            # Works if Dh is grid or scalar
            # --------------------------------
            Ri_scalar = float(Ri)
            if Ri_scalar > 0:
                Dh = Dh / (np.float64(1) + (np.float64(10) * Ri_scalar))
            else:
                Dh = Dh * (np.float64(1) - (np.float64(10) * Ri_scalar))

        # ----------------------------------------------------
        # NB! We currently assume that these are all equal.
        # ----------------------------------------------------
        self.Dn = Dn
        self.Dh = Dh
        self.De = Dh  ## (assumed equal)

    def update_sensible_heat_flux(self):
        """Physical constants
        ---------------------
        rho_air = 1.225d   ;[kg m-3, at sea-level]
        Cp_air  = 1005.7   ;[J kg-1 K-1]
        -----------------------------
        Compute sensible heat flux
        -----------------------------
        """  # noqa: D205
        LOG.debug("update_sensible_heat_flux")
        delta_T = self.T_air - self.T_surf
        self.Qh = (self.cfg.rho_air * self.cfg.Cp_air) * self.Dh * delta_T

    def update_saturation_vapor_pressure(self, MBAR=False, SURFACE=False):
        """Notes: Saturation vapor pressure is a function of temperature.
        #        T is temperature in Celsius.  By default, the method
        #        of Brutsaert (1975) is used, but if the SATTERLUND
        #        keyword is set then the method of Satterlund (1979) is
        #        used.  When plotted, they look almost identical.  See
        #        the compare_em_air_method routine in this file.
        #        Dingman (2002) uses the Brutsaert method.
        #        Liston (1995, EnBal) uses the Satterlund method.

        #        By default, the result is returned with units of kPa.
        #        Set the MBAR keyword for units of millibars.
        #        100 kPa = 1 bar = 1000 mbars
        #                => 1 kPa = 10 mbars
        # ----------------------------------------------------------------
        # NB!    Here, 237.3 is correct, and not a misprint of 273.2.
        #        See footnote on p. 586 in Dingman (Appendix D).
        # ----------------------------------------------------------------
        # Also see: topoflow.utils.met_utils.py   #################
        # ----------------------------------------------------------------
        # NOTE: If the temperature, T_air or T_surf, is constant in
        #       time, so that T_air_type or T_surf_type is in
        #       ['Scalar', 'Grid'], and if it has been initialized
        #       correctly, then there is no need to recompute e_sat.
        # ----------------------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_saturation_vapor_pressure")
        if SURFACE:
            #             HAVE_VAR   = hasattr(self, 'e_sat_surf'))
            #             T_CONSTANT = (self.T_surf_type in ['Scalar', 'Grid'])
            #             if (HAVE_VAR and T_CONSTANT): return
            T = self.T_surf
        else:
            #             HAVE_VAR   = hasattr(self, 'e_sat_air')
            #             T_CONSTANT = (self.T_air_type in ['Scalar', 'Grid'])
            #             if (HAVE_VAR and T_CONSTANT): return
            T = self.T_air

        if not (self.cfg.SATTERLUND):
            # ------------------------------
            # Use Brutsaert (1975) method
            # ------------------------------
            term1 = (np.float64(17.3) * T) / (T + np.float64(237.3))
            e_sat = np.float64(0.611) * np.exp(term1)  # [kPa]
        else:
            # -------------------------------
            # Use Satterlund (1979) method     #### DOUBLE CHECK THIS (7/26/13)
            # -------------------------------
            term1 = np.float64(2353) / (T + np.float64(273.15))
            e_sat = np.float64(10) ** (np.float64(11.4) - term1)  # [Pa]
            e_sat = e_sat / np.float64(1000)  # [kPa]

        # -----------------------------------
        # Convert units from kPa to mbars?
        # -----------------------------------
        if MBAR:
            e_sat = e_sat * np.float64(10)  # [mbar]

        if SURFACE:
            self.e_sat_surf = e_sat
        else:
            self.e_sat_air = e_sat

    def update_vapor_pressure_from_spHum_AirPre(self, SURFACE=False, MBAR=False):
        """
        Computes vapor pressure using specific humidity and total air pressure

        :param SURFACE: Flase or True
        :param MBAR: converts to mbar
        :return: None
        """
        LOG.debug("update_vapor_pressure_from_spHum_AirPre")
        e = self.Hum_sp * self.P_air / (self.cfg.eps + ((1 - self.cfg.eps) * self.Hum_sp))
        e = e / np.float64(1000)  # [kPa]

        if MBAR:
            e = e * np.float64(10)  # [mbar]

        if SURFACE:
            self.e_surf = e
        else:
            self.e_air = e

    def update_RH(self, SURFACE=False):
        """
        Updates relative humidity. Between [0, 1]

        :param SURFACE: False or True
        :return: None
        """
        LOG.debug("update_RH")
        if SURFACE:
            self.RH = self.e_surf / self.e_sat_surf
        else:
            self.RH = self.e_air / self.e_sat_air

    def update_vapor_pressure(self, SURFACE=False):
        """Notes: T is temperature in Celsius
        #        RH = relative humidity, in [0,1]
        #             by definition, it equals (e / e_sat)
        #        e has units of kPa.
        # ---------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_vapor_pressure")
        if SURFACE:
            e_sat = self.e_sat_surf
        else:
            e_sat = self.e_sat_air

        # -------------------------------------------------
        e = self.RH * e_sat

        if SURFACE:
            self.e_surf = e
        else:
            self.e_air = e

    def update_dew_point(self):
        """Notes:  The dew point is a temperature in degrees C and
        #         is a function of the vapor pressure, e_air.
        #         Vapor pressure is a function of air temperature,
        #         T_air, and relative humidity, RH.
        # -----------------------------------------------------------

        # -------------------------------------------
        # This formula needs e_air in kPa units.
        # See: Dingman (2002, Appendix D, p. 587).
        # 2023-09-01.  But it may contain a bug.
        # -------------------------------------------
        #         e_air_kPa = self.e_air / np.float64(10) # [mbar -> kPa]
        #         log_vp    = np.log( e_air_kPa )
        #         top = log_vp + np.float64(0.4926)
        #         bot = np.float64(0.0708) - (np.float64(0.00421) * log_vp)
        #         self.T_dew = (top / bot)    # [degrees C]
        # -------------------------------------------
        # This formula needs e_air in Pa units.
        # See: Dingman (2015, 3.2.5, p. 114).
        # -------------------------------------------
        #         e_air_Pa = self.e_air * 100 # [mbar -> Pa]
        #         self.T_dew = (top / bot)    # [degrees C]
        # -----------------------------------------------
        # This formula needs e_air in mbar units.
        # See: https://en.wikipedia.org/wiki/Dew_point
        # -----------------------------------------------
        """  # noqa: D205
        a = 6.1121  # [mbar]
        b = 18.678
        c = 257.14  # [deg C]
        # d = 234.5    # [deg C]
        # Floor e_air to a tiny positive number to avoid -inf/NaN on first step
        e_air_mbar = np.asarray(self.e_air, dtype="float64")
        log_term = np.log(e_air_mbar / a)
        if b == log_term:
            LOG.critical("Dewpoint calculation failed, (b - log_term) cannot equal 0")
        self.T_dew = c * log_term / (b - log_term)

        # log_term = np.log(self.e_air / a)
        # self.T_dew = c * log_term / (b - log_term)  # [deg C

    def update_T_surf(self):
        """
        # Estimate T_surf using T_dew (Raleigh et al. 2013).
        # Only run this function if T_surf is provided
        # as a scalar or grid so that it still varies in time
        # -------------------------------------------------
        """  # noqa: D205
        # -------------------------------------------------
        # If snow and/or ice are present,  T_surf cannot
        # exceed 0 deg C
        # -------------------------------------------------
        LOG.debug("update_T_surf")
        T_surf = np.where(
            ((self.h_snow > 0) | (self.h_ice > 0)),  # where snow or ice exists
            np.minimum(self.T_dew, np.float64(0)),  # T_surf is either T_dew or 0, whichever is lower
            self.T_dew,
        )  # everywhere else, T_surf = T_dew
        self.T_surf = T_surf

    def update_precipitable_water_content(self):
        """
        # Notes:  W_p is precipitable water content in centimeters,
        #         which depends on air temp and relative humidity.
        # ------------------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_precipitable_water_content")
        arg = np.float64(0.0614 * self.T_dew)
        self.W_p = np.float64(1.12) * np.exp(arg)  # [cm]

    def update_latent_heat_flux(self):
        """Notes:  Pressure units cancel out because e_air and
        #         e_surf (in numer) have same units (mbar) as
        #         p0 (in denom).
        # --------------------------------------------------------
        # According to Dingman (2002, p. 273), constant should
        # be 0.622 instead of 0.662 (Zhang et al., 2000).
        # --------------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_latent_heat_flux")
        const = self.cfg.latent_heat_constant
        factor = self.cfg.rho_air * self.cfg.Lv * self.De
        delta_e = self.e_air - self.e_surf
        self.Qe = factor * delta_e * (const / self.p0)

    def update_conduction_heat_flux(self):
        """Notes: The conduction heat flux from snow to soil for
        #        computing snowmelt energy, Qm, is close to zero.
        #        Currently, self.Qc = 0 in initialize().

        #        However, the conduction heat flux from surface and sub-
        #        surface for computing Qet is given by Fourier's Law,
        #        namely Qc = Ks(Tx - Ts)/x.

        #        All the Q's have units of W/m^2 = J/(m^2 s).
        # -----------------------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_conduction_heat_flux")
        pass  # Method not implemented in Topoflow: https://github.com/NOAA-OWP/topoflow/blob/db4d5877a32455beebe78edf5abe8d91df128665/topoflow/components/met_base.py#L1905

    def update_advection_heat_flux(self):
        """Notes: Currently, self.Qa = 0 in initialize().
        #        All the Q's have units of W/m^2 = J/(m^2 s).
        # ------------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_advection_heat_flux")
        pass  # Method not implemented in Topoflow: https://github.com/NOAA-OWP/topoflow/blob/db4d5877a32455beebe78edf5abe8d91df128665/topoflow/components/met_base.py#L1925

    def update_julian_day(self, time_units="seconds"):
        """Update the julian_day and year using pandas datetime.

        Cheap path (default): only advance time and compute decimal Julian day.
        Full solar geometry (True Solar Noon etc.) is computed only if
        self._skip_solar_geometry is False (i.e., when we *must* synthesize SW).
        """
        LOG.debug("update_julian_day")
        # -------------------------------------------------------
        # Compute the current datetime from start + offset
        # -------------------------------------------------------
        current_datetime = self.get_current_datetime(time_units=time_units)
        self.year = current_datetime.year

        # ----------------------------------
        # Update the *decimal* Julian day
        # ----------------------------------
        self.julian_day = (
            current_datetime.day_of_year
            - 1
            + current_datetime.hour / 24
            + current_datetime.minute / 1440
            + current_datetime.second / 86400
        )
        # -------------------------------------------------------
        # Cheap path: no expensive solar geometry if we have SW forcing
        # -------------------------------------------------------
        if getattr(self, "_skip_solar_geometry", True):
            # When not synthesizing shortwave, just set offsets to 0
            self.GMT_offset = np.float64(0.0)
            self.TSN_offset = np.float64(0.0)
            return

        # -----------------------------
        # Full geometry (rarely needed)
        # -----------------------------
        dec_part = self.julian_day - int(self.julian_day)
        clock_hour = dec_part * self.hours_per_day
        self.GMT_offset = solar.gmt_offset_hours(
            lat=self.cfg.lat, lon=self.cfg.lon, when_utc=self.start_datetime
        )
        solar_noon = solar.True_Solar_Noon(
            self.julian_day,
            self.cfg.lon,            # for USA region, lon is negative
            self.GMT_offset,         # time-zone offset from GMT/UTC in hours
            DST_offset=None,
            year=self.year,
        )
        self.TSN_offset = clock_hour - solar_noon  # [hours]

    def update_albedo(self, method="aging"):
        """Only use this routine if time varying albedo is not supplied as an input:
        ------------------------------------------------
        Dynamic albedo accounting for aging snow
        ------------------------------------------------
        (Rohrer and Braun 1994): alpha = alpha0 + K * e^(-nr)
        alpha = albedo
        alpha0 = minimum snowpack albedo (~0.4)
        K = constant (~0.44)
        n = number of days since last major snowfall, at least 3 cm over 3 days
        r = recession coefficient = 0.05 for temperatures < than
        0 deg C, 0.12 for temperatures > 0 deg C
        ------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_albedo")
        if method == "aging":
            albedo = self.albedo

            # Ensure 3-day buffer exists and matches dt-derived length
            secs_3days = 3 * 24 * 3600
            n_steps_3days = max(1, int(np.ceil(secs_3days / float(self.dt))))
            buf = getattr(self, "P_snow_3day_watershed", None)
            if buf is None or np.size(buf, axis=0) != n_steps_3days:
                self.P_snow_3day_watershed = np.zeros(n_steps_3days, dtype="float64")

            r = np.where((self.T_air > 0), 0.12, 0.05)
            K = 0.44
            alpha0 = 0.4

            # you can roll on different axes (time axis), shape of the DEM and time axis and roll on the time axis
            self.P_snow_3day_watershed = np.roll(self.P_snow_3day_watershed, -1, axis=0)
            ws_density_ratio = self.rho_H2O / self.rho_snow
            self.P_snow_3day_watershed[np.size(self.P_snow_3day_watershed, axis=0) - 1] = (
                self.P_snow * self.dt * ws_density_ratio
            )

            p_snow_3day_watershed_total = np.sum(
                self.P_snow_3day_watershed, axis=0
            )  # maybe multipy by timestep here # also make sure to only sum over time axis
            # self.p_snow_3day_watershed_total = p_snow_3day_watershed_total # if you want to output and make sure it's working properly

            self.n = np.where((p_snow_3day_watershed_total >= 0.03), 0, self.n)
            self.n = np.where((p_snow_3day_watershed_total < 0.03), self.n + self.days_per_dt, self.n)
            snow_albedo = alpha0 + K * np.exp(-self.n * r)

            albedo = np.where(
                (self.h_snow > 0),  # where snow exists
                snow_albedo,
                albedo,
            )
            albedo = np.where(
                ((self.h_snow == 0) & (self.h_ice > 0)),  # where ice exists without snow
                np.float64(0.3),
                albedo,
            )
            albedo = np.where(
                ((self.h_snow == 0) & (self.h_ice == 0)),  # where there is no snow or ice (tundra)
                np.float64(0.15),
                albedo,
            )
            self.albedo = albedo
        # ------------------------------------------------
        # Simple dynamic albedo depending on ice vs. snow vs. bare ground (tundra) using values from Dingman
        # ------------------------------------------------
        if method == "simple":
            albedo = self.albedo
            albedo = np.where(
                (self.h_snow > 0),  # where snow exists
                np.float64(0.75),
                albedo,
            )
            albedo = np.where(
                ((self.h_snow == 0) & (self.h_ice > 0)),  # where ice exists without snow
                np.float64(0.3),
                albedo,
            )
            albedo = np.where(
                ((self.h_snow == 0) & (self.h_ice == 0)),  # where there is no snow or ice (tundra)
                np.float64(0.15),
                albedo,
            )
            self.albedo = albedo

    def set_aspect_angle(self):
        """Set the aspect angle based on aspect parameter"""
        # ------------------------------------------------------
        # ---------------------------------------------------------
        alpha = (np.pi / 2) - self.cfg.aspect
        alpha = (self.twopi + alpha) % self.twopi
        # -----------------------------------------------
        is_nan = not np.isfinite(alpha)
        if is_nan:
            alpha = np.float64(0)

        self.alpha = alpha

    def set_slope_angle(self) -> None:
        """Set slope angle beta from slope magnitude; ensure within [0, pi/2]."""
        beta = np.arctan(self.slopes)
        beta = (self.twopi + beta) % self.twopi

        if not np.all(np.isfinite(beta)):
            beta = np.where(np.isfinite(beta), beta, np.float64(0))

        w_bad = np.logical_or(beta < 0, beta > (np.pi / 2))
        if np.any(w_bad):
            LOG.error("Some slope angles are out of range. Not updating beta for those cells.")
            beta = np.where(w_bad, np.float64(0), beta)

        self.beta = beta

    def update_net_shortwave_radiation(self):
        """Notes:  If time is before local sunrise or after local
        #         sunset then Qn_SW should be zero.
        # ---------------------------------------------------------
        # Compute Qn_SW for this time
        # --------------------------------
        """
        LOG.debug("update_net_shortwave_radiation")
        # Fast path: if SW_in (forcing) is available, use it directly.
        # Units are W m-2 and net shortwave = Kin * (1 - albedo) (Dingman 2015, Eq. 6B1.1)
        try:
            SW_in = np.asarray(self.SW_in, dtype="float64")
        except Exception:
            SW_in = None

        if SW_in is not None and np.all(np.isfinite(SW_in)):
            # stay in cheap mode (no solar geometry)
            self._skip_solar_geometry = True
            Qn_SW = SW_in * (1.0 - self.albedo)
            if np.ndim(self.Qn_SW) == 0:
                self.Qn_SW.fill(np.float64(Qn_SW))
            else:
                self.Qn_SW[:] = Qn_SW  # [W m-2]
            return

        # -----------------------------------------------------------
        # Fall back to analytic clear-sky model only if no forcing.
        # -----------------------------------------------------------
        # enable full solar geometry from now on
        self._skip_solar_geometry = False

        K_cs = solar.Clear_Sky_Radiation(
            self.cfg.lat,
            self.julian_day,
            self.W_p,
            self.TSN_offset,
            self.alpha,
            self.beta,
            self.albedo,
            self.cfg.dust_atten,
        )

        # net shortwave = Kin * (1 - albedo)
        Qn_SW = K_cs * (1 - self.albedo)

        if np.ndim(self.Qn_SW) == 0:
            self.Qn_SW.fill(Qn_SW)  #### (mutable scalar)
        else:
            self.Qn_SW[:] = Qn_SW  # [W m-2]


    def update_net_shortwave_radiation_old(self):
        """Notes:  If time is before local sunrise or after local
        #         sunset then Qn_SW should be zero.
        # ---------------------------------------------------------
        # Compute Qn_SW for this time
        # --------------------------------
        """  # noqa: D205

        K_cs = solar.Clear_Sky_Radiation(
            self.cfg.lat,
            self.julian_day,
            self.W_p,
            self.TSN_offset,
            self.alpha,
            self.beta,
            self.albedo,
            self.cfg.dust_atten,
        )

        # -------------------------------------------
        # 2024-03-06: Fix missing account for albedo
        # in net shortwave radiation calcs
        # Dingman 3rd Edition 2015 Eqn. 6B1.1:
        # net shortwave = Kin * (1-albedo)
        # -------------------------------------------
        Qn_SW = K_cs * (1 - self.albedo)

        if np.ndim(self.Qn_SW) == 0:
            self.Qn_SW.fill(Qn_SW)  #### (mutable scalar)
        else:
            self.Qn_SW[:] = Qn_SW  # [W m-2]

    def update_em_air(self):
        """NB!  The Brutsaert and Satterlund formulas for air
        emissivity as a function of air temperature are in
             close agreement; see compare_em_air_methods().
             However, we must pay close attention to whether
             equations require units of kPa, Pa, or mbar.

                    100 kPa = 1 bar = 1000 mbars
                       => 1 kPa = 10 mbars
        ---------------------------------------------------------
        NB!  Temperatures are assumed to be given with units
             of degrees Celsius and are converted to Kelvin
             wherever necessary by adding C_to_K = 273.15.

             RH = relative humidity [unitless]
        ---------------------------------------------------------
        NB!  I'm not sure about how F is added at end because
             of how the equation is printed in Dingman (2002).
             But it reduces to other formulas as it should.
        ---------------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_em_air")
        T_air_K = self.T_air + self.C_to_K

        if not (self.cfg.SATTERLUND):
            # -----------------------------------------------------
            # Brutsaert (1975) method for computing emissivity
            # of the air, em_air.  This formula uses e_air with
            # units of kPa. (From Dingman (2002, p. 196).)
            # See notes for update_vapor_pressure().
            # -----------------------------------------------------
            e_air_kPa = self.e_air / np.float64(10)  # [kPa]
            F = self.cfg.canopy_factor
            C = self.cfg.cloud_factor
            term1 = (1.0 - F) * 1.72 * (e_air_kPa / T_air_K) ** self.one_seventh
            term2 = 1.0 + (0.22 * C**2.0)
            em_air = (term1 * term2) + F
        else:
            # --------------------------------------------------------
            # Satterlund (1979) method for computing the emissivity
            # of the air, em_air, that is intended to "correct
            # apparent deficiencies in this formulation at air
            # temperatures below 0 degrees C" (see G. Liston)
            # Liston cites Aase and Idso(1978), Satterlund (1979)
            # --------------------------------------------------------
            e_air_mbar = self.e_air
            eterm = np.exp(-1 * (e_air_mbar) ** (T_air_K / 2016))
            em_air = 1.08 * (1.0 - eterm)

        # --------------------------
        # Update em_air, in-place
        # ---------------------------------------------------------
        # NB! Currently, em_air is always initialized as scalar,
        #     but could change to grid after assignment.  Must
        #     determine if scalar or grid in initialize().
        # ---------------------------------------------------------
        self.em_air = em_air
        #         if (np.ndim( self.em_air ) == 0):
        #             self.em_air.fill( em_air )   #### (mutable scalar)
        #         else:
        #             self.em_air[:] = em_air

    def update_net_longwave_radiation(self):
        """Notes: Net longwave radiation is computed using the
               Stefan-Boltzman law.  All four data types
               should be allowed (scalar, time series, grid or
               grid stack).

               Qn_LW = (LW_in - LW_out)
               LW_in   = em_air  * sigma * (T_air  + 273.15)^4
               LW_out  = em_surf * sigma * (T_surf + 273.15)^4

               Temperatures in [deg_C] must be converted to
               [K].  Recall that absolute zero occurs at
               0 [deg_K] or -273.15 [deg_C].

        ----------------------------------------------------------------
        First, e_air is computed as:
          e_air = RH * 0.611 * exp[(17.3 * T_air) / (T_air + 237.3)]
        Then, em_air is computed as:
          em_air = (1 - F) * 1.72 * [e_air / (T_air + 273.15)]^(1/7) *
                    (1 + 0.22 * C^2) + F
        ----------------------------------------------------------------
        Compute Qn_LW for this time
        --------------------------------
        """
        LOG.debug("update_net_longwave_radiation")
        T_surf_K = self.T_surf + self.C_to_K
        T_air_K = self.T_air + self.C_to_K
        LW_in = self.em_air * self.cfg.sigma * (T_air_K) ** 4.0
        LW_out = self.cfg.em_surf * self.cfg.sigma * (T_surf_K) ** 4.0

        # Account for reflection of atmospheric longwave by the surface
        LW_out += (1.0 - self.cfg.em_surf) * LW_in

        self.Qn_LW = LW_in - LW_out  # [W m-2]

    def update_net_longwave_radiation_old(self):
        """Notes: Net longwave radiation is computed using the
               Stefan-Boltzman law.  All four data types
               should be allowed (scalar, time series, grid or
               grid stack).

               Qn_LW = (LW_in - LW_out)
               LW_in   = em_air  * sigma * (T_air  + 273.15)^4
               LW_out  = em_surf * sigma * (T_surf + 273.15)^4

               Temperatures in [deg_C] must be converted to
               [K].  Recall that absolute zero occurs at
               0 [deg_K] or -273.15 [deg_C].

        ----------------------------------------------------------------
        First, e_air is computed as:
          e_air = RH * 0.611 * exp[(17.3 * T_air) / (T_air + 237.3)]
        Then, em_air is computed as:
          em_air = (1 - F) * 1.72 * [e_air / (T_air + 273.15)]^(1/7) *
                    (1 + 0.22 * C^2) + F
        ----------------------------------------------------------------
        Compute Qn_LW for this time
        --------------------------------
        """  # noqa: D205
        T_surf_K = self.T_surf + self.C_to_K

        T_air_K = self.T_air + self.C_to_K
        LW_in = self.em_air * self.cfg.sigma * (T_air_K) ** 4.0
        # LW_in = self.LW_in    # LW_in is already available from inputs
        LW_out = self.cfg.em_surf * self.cfg.sigma * (T_surf_K) ** 4.0

        # ----------------------------------------------------
        # 2023-08-29.  The next line was here before today,
        # and accounts for the amount of longwave radiation
        # from the air that is reflected from the surface.
        # See: https://daac.ornl.gov/FIFE/guides/
        #        Longwave_Radiation_UNL.html
        # It reduces the net longwave radiation.
        # ----------------------------------------------------
        LW_out += (1.0 - self.cfg.em_surf) * LW_in

        self.Qn_LW = LW_in - LW_out  # [W m-2]

        # --------------------------------------------------------------
        # Can't do this yet.  Qn_LW is always initialized grid now
        # but will often be created above as a scalar. (9/23/14)
        # --------------------------------------------------------------
        #         if (np.ndim( self.Qn_LW ) == 0):
        #             self.Qn_LW.fill( Qn_LW )   #### (mutable scalar)
        #         else:
        #             self.Qn_LW[:] = Qn_LW  # [W m-2]

    def update_net_energy_flux(self):
        """Notes: Q_sum is used by "snow_energy_balance.py".
        ------------------------------------------------------
               Qm    = energy used to melt snowpack (if > 0)
               Qn_SW = net shortwave radiation flux (solar)
               Qn_LW = net longwave radiation flux (air, surface)
               Qh    = sensible heat flux from turbulent convection
                       between snow surface and air
               Qe    = latent heat flux from evaporation, sublimation,
                       and condensation
               Qa    = energy advected by moving water (i.e. rainfall)
                       (ARHYTHM assumes this to be negligible; Qa=0.)
               Qc    = energy flux via conduction from snow to soil
                       (ARHYTHM assumes this to be negligible; Qc=0.)
               Ecc   = cold content of snowpack = amount of energy
                       needed before snow can begin to melt [J m-2]

               All Q's here have units of [W m-2].
               Are they all treated as positive quantities ?

               rho_air  = density of air [kg m-3]
               rho_snow = density of snow [kg m-3]
               Cp_air   = specific heat of air [J kg-1 K-1]
               Cp_snow  = heat capacity of snow [J kg-1 K-1]
                        = ???????? = specific heat of snow
               Kh       = eddy diffusivity for heat [m2 s-1]
               Ke       = eddy diffusivity for water vapor [m2 s-1]
               Lv       = latent heat of vaporization [J kg-1]
               Lf       = latent heat of fusion [J kg-1]
               ------------------------------------------------------
               Dn       = bulk exchange coeff for the conditions of
                          neutral atmospheric stability [m/s]
               Dh       = bulk exchange coeff for heat
               De       = bulk exchange coeff for vapor
               ------------------------------------------------------
               T_air    = air temperature [deg_C]
               T_surf   = surface temperature [deg_C]
               T_snow   = average snow temperature [deg_C]
               RH       = relative humidity [unitless] (in [0,1])
               e_air    = air vapor pressure at height z [mbar]
               e_surf   = surface vapor pressure [mbar]
               ------------------------------------------------------
               h_snow   = snow depth [m]
               z        = height where wind speed is uz [m]
               uz       = wind speed at height z [m/s]
               p0       = atmospheric pressure [mbar]
               T0       = snow temperature when isothermal [deg_C]
                          (This is usually 0.)
               z0_air   = surface roughness length scale [m]
                          (includes vegetation not covered by snow)
                          (Values from page 1033: 0.0013, 0.02 [m])
               kappa    = von Karman's constant [unitless] = 0.41
               dt       = snowmelt timestep [seconds]
        ----------------------------------------------------------------
        """  # noqa: D205
        LOG.debug("update_net_energy_flux")
        Q_sum = self.Qn_SW + self.Qn_LW + self.Qh + self.Qe + self.Qa + self.Qc  # [W m-2]

        if np.ndim(self.Q_sum) == 0:
            self.Q_sum.fill(Q_sum)  #### (mutable scalar)
        else:
            self.Q_sum = Q_sum  # [W m-2]

    def update_snow_meltrate(self):
        """Compute energy-balance meltrate
        # ------------------------------------------------------
        # Eccs is initialized by initialize_snow_cold_content().
        # ------------------------------------------------------
        # The following pseudocode only works for scalars but
        # is otherwise equivalent to that given below and
        # clarifies the logic:
        # ------------------------------------------------------
        #  if (Q_sum gt 0) then begin
        #      if ((Q_sum * dt) gt Eccs) then begin
        #          ;-------------------------------------------
        #          ; Snow is melting.  Use some of Q_sum to
        #          ; overcome Eccs, and remainder to melt snow
        #          ;-------------------------------------------
        #          Qm  = Q_sum - (Eccs/dt)
        #          Eccs = 0
        #          M   = (Qm / (rho_w * Lf))
        #      endif else begin
        #          ;------------------------------
        #          ; Snow is warming; reduce Eccs
        #          ;------------------------------
        #          Eccs = (Eccs - (Q_sum * dt))
        #          M   = 0d
        #      endelse
        #  endif else begin
        #      ;--------------------------------
        #      ; Snow is cooling; increase Eccs
        #      ;--------------------------------
        #      Eccs = Eccs - (Q_sum * dt)
        #      M   = 0d
        #  endelse
        # ---------------------------------------------------------
        # Q_sum = Qn_SW + Qn_LW + Qh + Qe + Qa + Qc    # [W m-2]
        # ---------------------------------------------------------

        # -----------------------------------------------
        # New approach; easier to understand
        # -----------------------------------------------
        # E_in  = energy input over one time step
        # E_rem = energy remaining in excess of Eccs
        # -----------------------------------------------
        """  # noqa: D205
        LOG.debug("update_snow_meltrate")
        E_in = self.Q_sum * self.dt
        E_rem = np.maximum(E_in - self.Eccs, np.float64(0))
        Qm = E_rem / self.dt  # [W m-2]
        M = Qm / (self.rho_H2O * self.Lf)  # [m/s]
        if np.size(self.SM) == 1:
            M = np.float64(M)  # avoid type change
            self.SM.fill(M)
        else:
            self.SM = M

    def update_ice_meltrate(self):
        """Compute energy-balance meltrate
        # ------------------------------------------------------
        # Ecci is initialized by initialize_ice_cold_content().
        # ------------------------------------------------------
        # The following pseudocode only works for scalars but
        # is otherwise equivalent to that given below and
        # clarifies the logic:
        # ------------------------------------------------------
        #  if (Q_sum gt 0) then begin
        #      if ((Q_sum * dt) gt Ecci) then begin
        #          ;-------------------------------------------
        #          ; Ice is melting.  Use some of Q_sum to
        #          ; overcome Ecci, and remainder to melt ice
        #          ;-------------------------------------------
        #          Qm  = Q_sum - (Ecci/dt)
        #          Ecci = 0
        #          M   = (Qm / (rho_w * Lf))
        #      endif else begin
        #          ;------------------------------
        #          ; Ice is warming; reduce Ecci
        #          ;------------------------------
        #          Ecci = (Ecci - (Q_sum * dt))
        #          M   = 0d
        #      endelse
        #  endif else begin
        #      ;--------------------------------
        #      ; Ice is cooling; increase Ecci
        #      ;--------------------------------
        #      Ecci = Ecci - (Q_sum * dt)
        #      M   = 0d
        #  endelse
        # ---------------------------------------------------------
        # Q_sum = Qn_SW + Qn_LW + Qh + Qe + Qa + Qc    # [W m-2]
        # ---------------------------------------------------------

        # -----------------------------------------------
        # New approach; easier to understand
        # -----------------------------------------------
        # E_in  = energy input over one time step
        # E_rem = energy remaining in excess of Ecci
        # -----------------------------------------------
        """  # noqa: D205
        E_in = self.Q_sum * self.dt
        E_rem = np.maximum(E_in - self.Ecci, np.float64(0))
        Qm = E_rem / self.dt  # [W m-2]

        if not hasattr(self, "previous_iwe"):
            self.previous_iwe = np.array(self.h_iwe, dtype="float64").copy()
        delta_iwe = np.asarray(self.h_iwe, dtype="float64") - np.asarray(self.previous_iwe, dtype="float64")

        M = Qm / (self.rho_H2O * self.Lf)  # [m/s]  TODO: m/hour? also shouldn't it be self.rho_ice?
        IM = np.maximum(M, np.float64(0))
        self.IM = np.where((self.h_swe == 0) & (self.previous_swe == 0), IM, np.float64(0))

        Ecci = np.maximum((self.Ecci - E_in), np.float64(0))

        Ecci = np.where((self.h_ice == 0), np.float64(0), Ecci)

        if np.size(self.Ecci) == 1:
            Ecci = np.float64(Ecci)  # avoid type change
            self.Ecci.fill(Ecci)
        else:
            self.Ecci[:] = Ecci

    def update_combined_meltrate(self):
        """We want to feed combined snow and ice melt to GIUH for
        runoff, so combine the IM and SM variables to create M_total (flux).
        Then convert flux [m s-1] to discharge [m3 s-1] using area.
        """
        # Always define M_total
        # NOTE: self.P (and hence self.P_rain) are in m s-1 already (converted in the setter).
        # Do NOT divide by 3600 here.
        M_total = self.IM + self.SM + self.P_rain  # [m s-1]

        # Convert flux (m/s) to depth per timestep: depth = flux * dt
        M_total_depth = M_total * self.dt

        # Update BMI output variable
        self._outputs.set_value("land_surface_water__runoff_depth", M_total_depth)

        # Persist flux (shape-safe)
        if isinstance(M_total, np.ndarray):
            self.M_total = M_total
        else:
            self.M_total = np.array([M_total], dtype="float64")

        # --- Compute discharge Q = flux * area (ALWAYS define Q) ---
        Q = self.M_total * self.da_m2  # [m3 s-1]

        # Ensure ndarray with float64 and shape (1,) for BMI
        if not isinstance(Q, np.ndarray):
            Q = np.array([Q], dtype="float64")
        elif Q.ndim == 0:
            Q = Q.reshape(1).astype("float64")
        else:
            Q = Q.astype("float64", copy=False)

        # Update BMI output
        self._outputs.set_value("channel_water_x-section__volume_flow_rate", Q)

    def enforce_max_snow_meltrate(self):
        """The max possible meltrate would be if all snow (given
        # by snow depth, h_snow, were to melt in the one time
        # step, dt.  Meltrate should never exceed this value.
        # Recall that: (h_snow / h_swe) = (rho_H2O / rho_snow)
        #                               = density_ratio > 1
        # So h_swe = h_snow / density_ratio.
        # Previous version had a bug; see below.
        # Now also using "out" keyword for "in-place".
        # -------------------------------------------------------
        SM_max = self.h_swe / self.dt
        self.SM = np.minimum(self.SM, SM_max, out=self.SM)  # [m s-1]

        # ------------------------------------------------------
        # Make sure meltrate is positive, while we're at it ?
        # Is already done by "Energy-Balance" component.
        # ------------------------------------------------------
        """  # noqa: D205
        # self.SM = np.maximum(self.SM, np.float64(0))
        max_SM = np.asarray(self.h_swe, dtype="float64") / float(self.dt)
        SM = np.asarray(self.SM, dtype="float64")
        SM = np.clip(SM, 0.0, max_SM)  # no negative, no over-melt
        if np.ndim(self.SM) == 0:
            self.SM.fill(float(SM))
        else:
            self.SM[:] = SM

    def enforce_max_ice_meltrate(self):
        """The max possible meltrate would be if all ice (given
        # by ice depth, h_ice, were to melt in the one time
        # step, dt.  Meltrate should never exceed this value.
        # -------------------------------------------------------
        """  # noqa: D205
        # IM_max = self.h_iwe / self.dt
        # self.IM = np.minimum(self.IM, IM_max, out=self.IM)  # [m s-1]

        # ------------------------------------------------------
        # Make sure meltrate is positive, while we're at it ?
        # Is already done by "Energy-Balance" component.
        # ------------------------------------------------------
        # np.maximum(self.IM, np.float64(0), out=self.IM)

        max_IM = np.asarray(self.h_iwe, dtype="float64") / float(self.dt)
        IM = np.asarray(self.IM, dtype="float64")
        IM = np.clip(IM, 0.0, max_IM)
        if np.ndim(self.IM) == 0:
            self.IM.fill(float(IM))
        else:
            self.IM[:] = IM

    def update_SM_integral(self):
        """Update mass total for SM, sum over all pixels
        # ------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.SM * self.da_m2 * self.dt)  # [m^3]
        self.vol_SM += np.sum(volume)  #### np.sum vs. sum ???

    def update_IM_integral(self):
        """Update mass total for IM, sum over all pixels
        # ------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.IM * self.da_m2 * self.dt)
        self.vol_IM += np.sum(volume)

    def update_snowfall_cold_content(self):
        """Copy previous timestep's CC and adjust from here
        ----------------------------------------------------
        For newly fallen snow, add cold content using
        the same equation used for initializing cold content,
        but use wet bulb temperature as T_snow for new snow:
        --------------------------------------------------------------------
        Wet bulb temp. equation from Stull 2011. Adapted from R code:
        https://github.com/SnowHydrology/humidity/blob/master/R/humidity.R
        --------------------------------------------------------------------
        """  # noqa: D205
        new_h_snow = (self.P_snow * self.dt) * self.ws_density_ratio
        Eccs = self.Eccs

        # ----------------------------------------------------
        # Prepare to adjust CC for land surface energy fluxes
        # ----------------------------------------------------
        E_in = self.Q_sum * self.dt  # [J m-2]
        T_wb = (
            self.T_air * np.arctan(0.151977 * ((self.RH + 8.313659) ** 0.5))
            + np.arctan(self.T_air + self.RH)
            - np.arctan(self.RH - 1.676331)
            + ((0.00391838 * (self.RH**1.5)) * np.arctan(0.023101 * self.RH))
            - 4.86035
        )

        del_T = self.T0_cc - T_wb

        # ----------------------------------------------------
        # Only where NEW snow has fallen (P_snow > 0), ADD
        # cold content for the new snow AND account for land
        # surface energy fluxes
        # ----------------------------------------------------
        Eccs = np.where(
            (self.P_snow > 0),
            (
                np.maximum(
                    (Eccs + ((self.rho_snow * self.Cp_snow) * new_h_snow * del_T) - E_in), np.float64(0)
                )
            ),
            Eccs,
        )  # make sure signs check out
        # Eccs = np.maximum(Eccs, np.float64(0)) # make sure signs check out

        if np.size(self.Eccs) == 1:
            Eccs = np.float64(Eccs)  # avoid type change
            self.Eccs.fill(Eccs)
        else:
            self.Eccs[:] = Eccs

    def update_snowpack_cold_content(self):
        """Copy the CC that was only adjusted in places WITH
        new snowfall before adjusting in places WITHOUT new
        snowfall
        -----------------------------------------------------
        """  # noqa: D205
        Eccs = self.Eccs

        E_in = self.Q_sum * self.dt  # [J m-2]
        # Eccs  = np.maximum((self.Eccs - E_in), np.float64(0))
        Eccs = np.where((self.P_snow <= 0), np.maximum((Eccs - E_in), np.float64(0)), Eccs)

        Eccs = np.where((self.h_snow == 0), np.float64(0), Eccs)

        if np.size(self.Eccs) == 1:
            Eccs = np.float64(Eccs)  # avoid type change
            self.Eccs.fill(Eccs)
        else:
            self.Eccs[:] = Eccs

    def extract_previous_swe(self):
        """Extract swe from previous timestep for use in
        toggling between ice/snow routines
        ------------------------------------------------
        """  # noqa: D205
        self.previous_swe = self.h_swe.copy()

    def update_swe(self):
        """The Meteorology component uses air temperature
        to compute P_rain (precip that falls as liquid) and
        P_snow (precip that falls as snow or ice) separately.
        P_snow = (self.P * (self.T_air <= 0))
        ----------------------------------------------------------
        Note: This method must be written to work regardless
        of whether P_rain and T are scalars or grids. (3/14/07)
        ------------------------------------------------------------
        If P or T_air is a grid, then h_swe and h_snow are grids.
        This is set up in initialize_computed_vars().
        ------------------------------------------------------------

        ------------------------------------------------
        Increase snow water equivalent due to snowfall
        ------------------------------------------------
        Meteorology and Channel components may have
        different time steps, but then self.P_snow
        will be a time-interpolated value.
        ------------------------------------------------
        """  # noqa: D205
        dh1_swe = self.P_snow * self.dt
        self.h_swe += dh1_swe

        # ------------------------------------------------
        # Decrease snow water equivalent due to melting
        # Note that SM depends partly on h_snow.
        # ------------------------------------------------
        # Compute potential melt depth for this timestep
        dh2_swe = self.SM * self.dt

        # Cap melt to available snow
        dh2_swe = np.minimum(dh2_swe, self.h_swe)

        # Calculate actual melt rate
        self.SM = dh2_swe / self.dt

        # Decrease SWE by actual melt
        self.h_swe -= dh2_swe

        # Ensure h_swe is non-negative
        np.maximum(self.h_swe, np.float64(0), self.h_swe)  # (in place)

    def update_iwe(self):
        """Decrease ice water equivalent due to melting
        ------------------------------------------------
        """  # noqa: D205
        # Compute potential melt depth for this timestep
        dh2_iwe = self.IM * self.dt

        # Cap melt to available ice
        dh2_iwe = np.minimum(dh2_iwe, self.h_iwe)

        # Calculate actual melt rate
        self.IM = dh2_iwe / self.dt
        
        # Decrease IWE by actual melt
        self.h_iwe -= dh2_iwe

        # Ensure h_iwe is non-negative
        np.maximum(self.h_iwe, np.float64(0), self.h_iwe)  # (in place)

    def update_ws_density_ratio(self):
        """Return if density_ratio is constant in time.
        -----------------------------------------------
        """  # noqa: D205
        density_ratio = self.cfg.rho_H2O / self.cfg.rho_snow

        # -------------------------------------
        # Save updated density ratio in self
        # -------------------------------------
        if np.ndim(self.ws_density_ratio) == 0:
            density_ratio = np.float64(density_ratio)  ### (from 0D array to scalar)
            self.ws_density_ratio.fill(density_ratio)  ### (mutable scalar)
        else:
            self.ws_density_ratio[:] = density_ratio

    def update_wi_density_ratio(self):
        """Return if density_ratio is constant in time.
        -----------------------------------------------
        """  # noqa: D205
        density_ratio = self.cfg.rho_H2O / self.cfg.rho_ice

        # -------------------------------------
        # Save updated density ratio in self
        # -------------------------------------
        if np.ndim(self.wi_density_ratio) == 0:
            density_ratio = np.float64(density_ratio)  ### (from 0D array to scalar)
            self.wi_density_ratio.fill(density_ratio)  ### (mutable scalar)
        else:
            self.wi_density_ratio[:] = density_ratio

    def update_swe_integral(self):
        """Update mass total for water in the snowpack,
        sum over all pixels.
        ------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.h_swe * self.cfg.da)  # [m^3]
        if np.size(volume) == 1:
            self.vol_swe += volume * self.rti.n_pixels
        else:
            self.vol_swe += np.sum(volume)

    def update_iwe_integral(self):
        """Update mass total for water in the ice column,
        sum over all pixels.
        ------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.h_iwe * self.cfg.da)  # [m^3]
        if np.size(volume) == 1:
            self.vol_iwe += volume * self.rti.n_pixels
        else:
            self.vol_iwe += np.sum(volume)

    def extract_previous_snow_depth(self):
        """Extract swe from previous timestep for use in
        toggling between ice/snow routines
        ------------------------------------------------
        """  # noqa: D205
        self.previous_h_snow = self.h_snow.copy()

    def update_snow_depth(self):
        """The Meteorology component uses air temperature
        to compute P_rain (precip that falls as liquid) and
        P_snow (precip that falls as snow or ice) separately.
        P_snow = (self.P * (self.T_air <= 0))
        ----------------------------------------------------------
        Note: This method must be written to work regardless
        of whether P_rain and T are scalars or grids.
        ------------------------------------------------------------
        If P or T_air is a grid, then h_swe and h_snow are grids.
        This is set up in initialize_computed_vars().
        ------------------------------------------------------------
        Note that for a region of area, A:
            rho_snow = (mass_snow / (h_snow * A))
            rho_H2O  = (mass_H20  / (h_swe * A))
        Since mass_snow = mass_H20 (for SWE):
            rho_snow * h_snow = rho_H2O * h_swe
            (h_snow / h_swe)  = (rho_H2O / rho_snow)
             h_snow = h_swe * density_ratio
        Since liquid water is denser than snow:
             density_ratio > 1 and
             h_snow > h_swe
        self.density_ratio = (self.rho_H2O / self.rho_snow)
        rho_H2O is for liquid water close to 0 degrees C.
        ------------------------------------------------------------

        -------------------------------------------------
        Change snow depth due to melting or falling snow
        -------------------------------------------------
        This assumes that update_swe() is called
        before update_snow_depth().
        -------------------------------------------
        """  # noqa: D205
        h_snow = self.h_swe * self.ws_density_ratio

        if np.ndim(self.h_snow) == 0:
            h_snow = np.float64(h_snow)  ### (from 0D array to scalar)
            self.h_snow.fill(h_snow)  ### (mutable scalar)
        else:
            self.h_snow[:] = h_snow

    def update_ice_depth(self):
        """Change ice depth due to melting
        ---------------------------------
        This assumes that update_iwe() is called
        before update_ice_depth().
        -------------------------------------------
        """  # noqa: D205
        h_ice = self.h_iwe * self.wi_density_ratio
        if np.ndim(self.h_ice) == 0:
            h_ice = np.float64(h_ice)
            self.h_ice.fill(h_ice)
        else:
            self.h_ice[:] = h_ice

    def update_total_snowpack_water_volume(self):
        """Compute the total volume of water stored
               in the snowpack for all grid cells in the DEM.
               Use this in the final mass balance reporting.
               (2023-08-31)
        --------------------------------------------------------
        Note:  This is called from initialize() & finalize().
        --------------------------------------------------------

        ----------------------------------------------------
        Update total volume of liquid water stored in the
        current snowpack, sum over all grid cells but no
        integral over time.  (2023-08-31)
        ----------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.h_swe * self.da)  # [m^3]
        if np.size(volume) == 1:
            vol_swe = volume * self.rti.n_pixels
        else:
            ## volume[ self.edge_IDs ] = 0.0  # (not needed)
            vol_swe = np.sum(volume)

        self.vol_swe.fill(vol_swe)

    #   update_total_snowpack_water_volume()
    # -------------------------------------------------------------------
    def update_total_ice_water_volume(self):
        """Compute the total volume of water stored
               in the ice for all grid cells in the DEM.
               Use this in the final mass balance reporting.
               (2023-08-31)
        --------------------------------------------------------
        Note:  This is called from initialize() & finalize().
        --------------------------------------------------------

        ----------------------------------------------------
        Update total volume of liquid water stored in the
        current ice, sum over all grid cells but no
        integral over time.  (2023-08-31)
        ----------------------------------------------------
        """  # noqa: D205
        volume = np.float64(self.h_iwe * self.cfg.da)  # [m^3]
        vol_iwe = np.sum(volume)

        self.vol_iwe.fill(vol_iwe)

    def get_component_name(self) -> str:
        """Name of this BMI module component.

        Returns
        -------
            str: Model Name
        """
        return "Topoflow-Glacier"

    def get_output_item_count(self) -> int:
        """Returns the number of output state variables"""
        return len(self._outputs)

    def get_input_var_names(self) -> list[str]:
        """Return BMI input variable names that NGen can set."""
        LOG.debug("get_input_var_names")
        # The Context you build from _dynamic_input_vars already has the names.
        return list(self._dynamic_inputs.names())

    def get_output_var_names(self) -> list[str]:
        """Return BMI output variable names that NGen can read."""
        LOG.debug("get_output_var_names")
        return list(self._outputs.names())

    def get_value(self, name: str, dest: np.ndarray) -> np.ndarray:
        """BMI get_value: copy variable 'name' into provided 'dest' array."""
        # Prefer outputs first, then inputs, so discharge/melt are readable
        dest[:] = self.get_value_ptr(name)
        return dest

    def set_value(self, name: str, values) -> None:
        """BMI set_value: assign into BMI variable 'name' from 'values' array."""
        if name == Serialization.CREATE:
            self._serialize()
            return
        elif name == Serialization.STATE:
            self._deserialize(values)
            return
        elif name == Serialization.FREE:
            self._free_serialized()
            return
        elif name == Serialization.RESET:
            self._reset_time()
            return

        arr = np.asarray(values, dtype="float64").reshape(-1)

        if name == "ngen_realization_start_time":
            self._ngen_realization_start_time = float(arr[0])
            self._dynamic_inputs.set_value(name, arr)
            self._try_apply_ngen_realization_time()
            return

        if name == "ngen_realization_end_time":
            self._ngen_realization_end_time = float(arr[0])
            self._dynamic_inputs.set_value(name, arr)
            self._try_apply_ngen_realization_time()
            return

        if name == "ngen_realization_dt":
            self._ngen_realization_dt = float(arr[0])
            self._dynamic_inputs.set_value(name, arr)
            self._try_apply_ngen_realization_time()
            return

        if name == "T_rain_snow":
            # Set calibratable parameters
            try:
                self._calibs.set_value(name, arr)
            except Exception:
                pass
            return

        if name in {
            "land_surface_wind__x_component_of_velocity",
            "U2D",
            "atmosphere_wind__x_component_of_velocity"
        }:
            # Accept U-component (m s-1)
            self._dynamic_inputs.set_value("land_surface_wind__x_component_of_velocity", arr)
            self._recompute_wind_speed()
            return

        if name in {
            "land_surface_wind__y_component_of_velocity",
            "V2D",
            "atmosphere_wind__y_component_of_velocity"
        }:
            # Accept V-component (m s-1)
            self._dynamic_inputs.set_value("land_surface_wind__y_component_of_velocity", arr)
            self._recompute_wind_speed()
            return

        if name in {"wind_speed_UV", "land_surface_wind__speed"}:
            self.uz = values
            return

        if name == "atmosphere_water__liquid_equivalent_precipitation_rate":
            # Convert mm h-1 -> m s-1
            vals_mps = arr / 3_600_000.0
            try:
                self._dynamic_inputs.set_value(name, vals_mps)
            except Exception:
                pass
            return

        # Pass-through for other known dynamic inputs
        try:
            if name in self._dynamic_inputs:
                self._dynamic_inputs.set_value(name, arr)
                return
        except Exception:
            # If context lookup fails, continue to outputs/raise
            pass

        # Pass-through for other known calib inputs
        try:
            if name in self._calibs:
                self._calibs.set_value(name, arr)
                return
        except Exception:
            # If context lookup fails, continue to outputs/raise
            pass

        # Allow writing to outputs if caller uses set_value on them
        try:
            if name in self._outputs:
                self._outputs.set_value(name, arr)
                return
        except Exception:
            pass

        raise KeyError(f"Unknown BMI variable name: {name}")

    def set_value_old(self, name: str, values) -> None:
        """BMI set_value: assign into BMI variable 'name' from 'values' array."""
        arr = np.asarray(values, dtype="float64").reshape(-1)

        if name == "land_surface_wind__x_component_of_velocity":
            # store component and recompute speed
            self._wind_u = float(arr[0])
            self._recompute_wind_speed()
            # Keep internal mirrors if you expose them via Context elsewhere
            try:
                self._dynamic_inputs.set_value("land_surface_wind__x_component_of_velocity", arr)
            except Exception:
                pass
            return

        if name == "land_surface_wind__y_component_of_velocity":
            self._wind_v = float(arr[0])
            self._recompute_wind_speed()
            try:
                self._dynamic_inputs.set_value("land_surface_wind__y_component_of_velocity", arr)
            except Exception:
                pass
            return

        # Legacy compatibility: some scripts used to call this
        if name == "wind_speed_UV":
            # Accept, but treat as derived speed only. NGen should not send this.
            self.uz = arr
            # Accept scalar or 1-element array, keep internal cache in sync
            # v = float(np.asarray(values).reshape(-1)[0])
            # self._wind_speed = v
            # Also reflect it in the dynamic-input context so BMI reads work:
            # self._dynamic_inputs.set_value("wind_speed_UV", np.array([v], dtype="float64"))
            return

        # Handle inputs (forcing) that NGen writes
        if name in self._dynamic_inputs:
            # Special-case units you defined in _dynamic_input_vars
            # P (mm h-1) -> internal m s-1 via self.mmph_to_mps (set in initialize)
            if name == "atmosphere_water__liquid_equivalent_precipitation_rate":
                self.P = arr  # property handles mm/h -> m/s
                return
            # Everything else is stored as-is
            self._dynamic_inputs.set_value(name, arr)
            return

        # Handle outputs (rare for BMI but allowed)
        if name in self._outputs:
            self._outputs.set_value(name, arr)
            return

        raise KeyError(f"Unknown BMI variable name: {name}")

    def _warn_if_no_initial_storage(self) -> None:
        try:
            h_swe0 = float(np.asarray(self._outputs.value("snowpack__liquid-equivalent_depth")).reshape(-1)[0])
            h_snow0 = float(np.asarray(self._outputs.value("snowpack__depth")).reshape(-1)[0])
            h_iwe0 = float(np.asarray(self._outputs.value("glacier__liquid_equivalent_depth")).reshape(-1)[0])
            h_ice0 = float(np.asarray(self._outputs.value("glacier_ice__thickness")).reshape(-1)[0])
        except Exception:
            return
        if (h_swe0 <= 0.0) and (h_snow0 <= 0.0) and (h_iwe0 <= 0.0) and (h_ice0 <= 0.0):
            LOG.warning(
                "Initial SWE/ice are all zero (h0_swe=h0_snow=h0_iwe=h0_ice=0). "
                "Without rainfall in forcing, discharge will remain 0."
            )
    def get_value_at_indices(self, name: str, dest: np.ndarray, inds: np.ndarray) -> np.ndarray:
        LOG.debug(f"get_value_at_indices: {name}")
        a_inds = np.asarray(inds, dtype=int)

        if hasattr(self, "_outputs") and name in self._outputs:
            return self._outputs.value_at_indices(name, dest, a_inds)
        if hasattr(self, "_dynamic_inputs") and name in self._dynamic_inputs:
            return self._dynamic_inputs.value_at_indices(name, dest, a_inds)
        if hasattr(self, "_calibs") and name in self._calibs:
            return self._calibs.value_at_indices(name, dest, a_inds)

        raise KeyError(f"Variable not found: {name}")

    def set_value_at_indices(self, name: str, inds: np.ndarray, src: np.ndarray) -> None:
        LOG.debug(f"set_value_at_indices: {name}")
        a_inds = np.asarray(inds, dtype=int)
        a_src = np.asarray(src)

        if hasattr(self, "_dynamic_inputs") and name in self._dynamic_inputs:
            self._dynamic_inputs.set_value_at_indices(name, a_inds, a_src)
            return
        if hasattr(self, "_calibs") and name in self._calibs:
            self._calibs.set_value_at_indices(name, a_inds, a_src)
            return
        if hasattr(self, "_outputs") and name in self._outputs:
            self._outputs.set_value_at_indices(name, a_inds, a_src)
            return

        raise KeyError(f"Variable not found: {name}")

    def get_value_ptr(self, name: str) -> NDArray:
        """Gets value in native form if exists in inputs or outputs."""

        if name == "wind_speed_UV" or name == "land_surface_wind__speed":
            return self.uz
        if name == Serialization.STATE:
            return self._serialized
        if Serialization.dtype(name) is not None:
            return self._serialized_size

        return first_containing(name, self._outputs, self._dynamic_inputs, self._calibs).value(name)

    def get_var_itemsize(self, name: str) -> int:
        """Size, in bytes, of a single element of the variable name

        Args:
            name (str): variable name

        Returns
        -------
            int: number of bytes representing a single variable of @p name
        """
        if Serialization.dtype(name) is not None:
            return Serialization.dtype(name).itemsize
        return self.get_value_ptr(name).itemsize

    def get_var_nbytes(self, name: str) -> int:
        """Size, in nbytes, of a single element of the variable name

        Args:
            name (str): Name of variable.

        Returns
        -------
            int: Size of data array in bytes.
        """
        if Serialization.dtype(name) is not None:
            if name == Serialization.STATE:
                return self._serialized.nbytes
            return Serialization.dtype(name).itemsize
        return self.get_value_ptr(name).nbytes

    def get_var_type(self, name: str) -> str:
        """Data type of variable.

        Args:
            name (str): Name of variable.

        Returns
        -------
            str: Data type.
        """
        if Serialization.dtype(name) is not None:
            return str(Serialization.dtype(name))
        return str(self.get_value_ptr(name).dtype)

    def get_current_datetime(self, time_units: str = "seconds"):
        """
        Return current datetime from immutable realization start + current BMI time.

        Do not mutate self.start_datetime here. The realization start time must remain
        fixed so repeated update() calls do not drift the model clock.
        """
        if not isinstance(self.start_datetime, pd.Timestamp):
            self.start_datetime = pd.to_datetime(self.start_datetime)

        current_seconds = float(self.get_current_time())
        self.current_datetime = self.start_datetime + pd.to_timedelta(current_seconds, unit="s")

        return self.current_datetime

    def get_var_units(self, name: str) -> str:
        units = {
            # Inputs (advertised)
            "atmosphere_water__liquid_equivalent_precipitation_rate": "mm h-1",
            "land_surface_air__temperature": "degC",
            "land_surface_radiation~incoming~longwave__energy_flux": "W m-2",
            "land_surface_radiation~incoming~shortwave__energy_flux": "W m-2",
            "land_surface_air__pressure": "Pa",
            "atmosphere_air_water~vapor__relative_saturation": "1",
            "atmosphere_bottom_air_water-vapor__relative_saturation": "1",
            "wind_speed_UV": "m s-1",
            "land_surface_wind__speed": "m s-1",
            "land_surface_wind__x_component_of_velocity": "m s-1",
            "land_surface_wind__y_component_of_velocity": "m s-1",

            # Outputs / states
            "snowpack__melt_volume_flux": "m s-1",
            "glacier_ice__melt_volume_flux": "m s-1",
            "land_surface_water__runoff_volume_flux": "m s-1",
            "land_surface_water__runoff_depth": "m",
            "snowpack__depth": "m",
            "glacier_ice__thickness": "m",
            "snowpack__liquid-equivalent_depth": "m",
            "snowpack__liquid-equivalent_mass_per_area": "kg m-2",
            "glacier__liquid_equivalent_depth": "m",
            "precipitation_rate": "mm s-1",
            "channel_water_x-section__volume_flow_rate": "m3 s-1",

            # New outputs
            "atmosphere_water__snowfall_leq-volume_flux": "mm s-1",
            "snowpack__domain_time_integral_of_melt_volume_flux": "mm",
            "land_surface__temperature": "K",
        }
        try:
            return units[name]
        except KeyError:
            raise ValueError(f"Unknown variable for units: {name!r}")

    def get_var_itemcount(self, name: str) -> int:
        """
        Number of values for the given variable.
        Return 1 for scalar/site-mean variables. If/when you expose gridded
        variables, compute from the variable's grid id and grid size.
        """
        # Simple scalar case:
        return 1
        # Robust (enable later if you add gridded vars):
        # grid_id = self.get_var_grid(name)
        # return int(self.get_grid_size(grid_id))

    def get_input_item_count(self) -> int:
        """Aggregate item count across all input vars."""
        return int(sum(self.get_var_itemcount(v) for v in self.get_input_var_names()))

    def get_output_item_count(self) -> int:
        """Aggregate item count across all output vars."""
        return int(sum(self.get_var_itemcount(v) for v in self.get_output_var_names()))

    def _finalize_time_bounds(self) -> None:
        """
        Derive numeric end-of-run bounds for the BMI clock based on configured
        start/end datetimes and the fixed time step (seconds). This prepares:
          - self._n_steps        : integer number of update() steps
          - self._run_end_time_s : exclusive end time in seconds
        """
        # Parse start/end datetimes from the parsed YYYYMMDDHH fields
        self.start_datetime = pd.to_datetime(
            solar.get_datetime_str(self.start_year, self.start_month, self.start_day, self.start_hour, 0, 0)
        )
        self.end_datetime = pd.to_datetime(
            solar.get_datetime_str(self.end_year, self.end_month, self.end_day, self.end_hour, 0, 0)
        )

        # Compute exclusive end time as an integer number of dt steps
        total_seconds = float((self.end_datetime - self.start_datetime).total_seconds())
        self._timestep_size_s = float(self.dt)
        # The adapter may request exactly the exclusive end; keep a tiny slack
        self._n_steps = int(np.ceil(total_seconds / self._timestep_size_s - 1e-12))
        self._run_end_time_s = float(self._n_steps) * self._timestep_size_s

    def _serialize(self):
        """Create a serialized copy of the current model state needed to reload a prior timestep or hot start the model."""
        serializable = {
            "dynamic_inputs": self._dynamic_inputs.serializable(),
            "outputs": self._outputs.serializable(),
            "attr": { attr: getattr(self, attr) for attr in self._serializable_attr() }
        }
        serialized = pickle.dumps(serializable)
        self._serialized = np.array(bytearray(serialized), dtype=self._serialized.dtype)
        self._serialized_size[0] = self._serialized.nbytes

    def _deserialize(self, arr: NDArray):
        """Load a prior model state from a numpy array of bytes."""
        deserialized = pickle.loads(bytes(arr))
        self._dynamic_inputs.load_serialized(deserialized["dynamic_inputs"])
        self._outputs.load_serialized(deserialized["outputs"])
        for attr, value in deserialized["attr"].items():
            setattr(self, attr, value)
        self._free_serialized()

    def _free_serialized(self):
        """Create a new instance of the serialization array, letting the GC free any prior instance."""
        self._serialized = np.array([], Serialization.dtype(Serialization.STATE))
        self._serialized_size = np.array([0], dtype=Serialization.dtype(Serialization.SIZE))

    def _reset_time(self):
        """Reset the current time-based properties to the default value after `initialize` was run.\n
        This includes both attributes that store time information and BMI values that represent a sum from all timesteps."""
        for attr in self._time_reset_attr():
            value = getattr(self, attr)
            if isinstance(value, int):
                setattr(self, attr, 0)
            elif isinstance(value, float):
                setattr(self, attr, 0.0)
            else: # assume it's a numpy array
                value[:] = 0.0

    def _serializable_attr(self):
        return [
            "albedo", # updates based on itself
            "P_snow_3day_watershed", # updates based on itself
            "n", # updates based on itself
            "em_air", # updates based on itself,
            "Eccs", # updates from prior ws_density_ratio
            "Ecci", # updates based on itself
            "uz", # pseudo-dynamic input
        ] + self._time_reset_attr()

    def _time_reset_attr(self):
        return [
            "vol_P", # sum between updates
            "vol_PR", # sum between updates
            "vol_PS", # sum between updates
            "vol_IM", # sum between updates
            "vol_SM", # sum between updates
            "_timestep",
            "_t_index",
        ]

def first_containing(name: str, *states: Context) -> Context:
    """Return the first `State` object containing `name` in `states`. Otherwise, raise `KeyError`."""
    for state in states:
        if name in state:
            return state
    raise KeyError(f"Unknown BMI variable name: {name!s}")

class Serialization:
    STATE = "serialization_state"
    SIZE = "serialization_size"
    CREATE = "serialization_create"
    FREE = "serialization_free"
    RESET = "reset_time"

    @staticmethod
    def dtype(method: str):
        if method == Serialization.STATE:
            return np.dtype(np.uint8)
        if method == Serialization.RESET:
            return np.dtype(np.double)
        if (
            method == Serialization.CREATE
            or method == Serialization.SIZE
            or method == Serialization.FREE
        ):
            return np.dtype(np.uint64)
        return None

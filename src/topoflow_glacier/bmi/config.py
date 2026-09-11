from pydantic import BaseModel, ConfigDict, Field

__all__ = ["TopoflowGlacierConfig"]


class TopoflowGlacierConfig(BaseModel):
    """Validates the topoflow glacier config file"""

    # -----------------------
    # Required configuration
    # -----------------------
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_prefix: str = Field(description="File prefix for the study site")
    forcing_file: str = Field(description="The forcing .csv file to be used")
    dt: int = Field(ge=0, description="Timestep for snowmelt process [hour]")
    start_time: str | None = Field(
        default=None,
        description="(Optional) Start time [YYYYMMDDHH]. Ignored when provided via ngen realization."
    )

    end_time: str | None = Field(
        default=None,
        description="(Optional) End time [YYYYMMDDHH]. Ignored when provided via ngen realization."
    )
    da: float = Field(description="The drainage area of the modeled reach [km2]")
    slope: float = Field(description="The slope of the catchment [m km-1]")
    lat: float = Field(description="Latitude of the centroid of the catchment")
    lon: float = Field(description="Longitude of the centroid of the catchment")
    h0_snow: float = Field(description="Initial depth of snow [m]")
    h0_ice: float = Field(description="Initial depth of ice [m]")
    h0_swe: float = Field(description="Initial depth of snow water equivalent (SWE) [m]")
    h0_iwe: float = Field(description="Initial depth of ice water equivalent (IWE) [m]")
    elev: float = Field(description="Average watershed elevation [m]")
    T_rain_snow: float = Field(1.0, description="Degree-day temperature parameter (degree C)")
    aspect: float = Field(0.0, description="Aspect angle for each catchment")
    dust_atten: float = Field(
        0.08, ge=0.0, le=0.2, description="Atmosphere aerosol dust reduction of transmittance"
    )
    canopy_factor: float = Field(0.0, ge=0.0, le=1.0, description="canopy factor tha masks solar radiation")
    cloud_factor: float = Field(0.0, ge=0.0, le=1.0, description="cloud percentage")

    # -----------------------
    # Physical constants
    # -----------------------
    rho_air: float = Field(1.2614, description="Density of air [kg/m^3]")
    rho_snow: float = Field(50.0, description="Density of snow [kg/m^3]")
    rho_ice: float = Field(917.0, description="Density of ice [kg/m^3]")
    rho_H2O: float = Field(1000.0, description="Density of water [kg/m^3]")
    h_active_layer: float = Field(0.125, description="Thickness of active ice layer [m]")
    T0: float = Field(-0.2, description="Reference temperature [deg C]")
    Cp_air: float = Field(1005.7, description="Specific heat capacity of air [J/(kg * K)]]")
    Cp_ice: float = Field(2060.0, description="Specific heat capacity of ice [J/(kg * K)]]")
    Cp_snow: float = Field(2090.0, description="Specific heat capacity of snow [J/(kg * K)]]")
    g: float = Field(9.81, description="Gravity  [m/s2]")
    Lf: float = Field(334000.0, description="Latent heat of fusion [J kg-1]")
    eps: float = Field(0.622, description="Ratio of gas constant [unitless]")
    kappa: float = Field(0.408, description="Von Karman constant [unitless]")
    latent_heat_constant: float = Field(0.622, description="According to Dingman (2002, p. 273)")
    Lv: float = Field(2500000, description="Latent heat of vaporize [J kg-1]")
    sigma: float = Field(5.67 * 10 ** (-8), description="Stefan-Boltzman constant [W m-2 K-4]")

    sea_level_p0: float = Field(default=101325.0, description="Sea-level standard pressure [Pa]")
    sea_level_T0: float = Field(default=288.15, description="Sea-level standard temperature [K]")
    T_lapse_rate: float = Field(default=0.0065, description="Temperature lapse rate [K/m]")
    uni_gas_const: float = Field(default=8.3144598, description="  #Universal gas constant [J/mol/K]")
    M_mass_air: float = Field(default=0.0289644, description="  #molar mass of dry air [kg/mol]")

    # -----------------------
    # Glacier dynamics
    # -----------------------
    min_glacier_thick: float = Field(default=1.0, description="Minimum glacier thickness [m]")
    glens_A: float = Field(default=2.142e-16, description="Glen's Law exponent [Pa^-3 s^-1]")
    B: float = Field(default=0.0012, description="Flow law parameter [m / (Pa * yr)], see MacGregor (2000)")
    char_sliding_vel: float = Field(default=10.0, description="Characteristic sliding velocity [m/yr]")
    char_tau_bed: float = Field(default=100000.0, description="Characteristic shear stress at the bed [Pa]")
    depth_to_water_table: float = Field(
        default=20.0, description="Distance from ice surface to water table [m]"
    )
    max_float_fraction: float = Field(default=80.0, description="Limits the water level in ice [percent]")
    Hp_eff: float = Field(default=20.0, description="Effective pressure [m] of water")
    init_ELA: float = Field(default=3350.0, description="Initial Equilibrium Line Altitude [m]")
    ELA_step_size: float = Field(default=-10.0, description="ELA step size [m]")
    ELA_step_interval: float = Field(default=500.0, description="ELA step interval [m]")
    grad_Bz: float = Field(default=0.01, description="Mass balance gradient in z [m/yr/m]")
    max_Bz: float = Field(default=2.0, description="Maximum allowed mass balance [m/yr]")
    spinup_time: float = Field(default=200.0, description="Spinup time [years]")
    sea_level: float = Field(default=-100.0, description="Sea level [m]")
    z0_air: float = Field(default=0.01, ge=0.0001, le=0.1, description="Surface roughness length [m]")
    em_surf: float = Field(default=0.985, ge=0.9, le=1, description="Surface roughness length [m]")

    geothermal_heat_flux: float = Field(default=1575000.0, description="Geothermal heat flux [(J/year)/m^2]")
    geothermal_gradient: float = Field(default=-0.0255, description="Geothermal gradient [deg_C/m]")

    # -----------------------
    # “Missing CFG” legacy toggles — make them real fields with defaults
    ## comes from def set_missing_cfg_options(self)
    # check this link for original code:
    # https://github.com/NOAA-OWP/topoflow/blob/db4d5877a32455beebe78edf5abe8d91df128665/topoflow/components/met_base.py#L856
    # -----------------------
    PRECIP_ONLY: bool = Field(False, description="If True, only precip is used (legacy toggle)")
    P_factor: float = Field(1.0, description="Precip multiplier (legacy toggle)")

    # E.g. restrict choices with Literal; expand as needed
    # T_rain_snow_type: Literal['Scalar', 'Grid', 'TimeSeries'] = Field(
    #     'Scalar', description="How to interpret T_rain_snow"
    # )

    SATTERLUND: bool = Field(False, description="Use Satterlund method for e_air/em_air")
    # NGEN_CSV: bool = Field(False, description="Read precip series from NextGen CSV")

    # start_year: Optional[int] = Field(None, description="Start year (optional; derived if omitted)")
    #
    # # Save-output toggles
    # SAVE_QSW_GRIDS: bool = Field(False, description="Save shortwave radiation grids")
    # SAVE_QLW_GRIDS: bool = Field(False, description="Save longwave radiation grids")
    # SAVE_TSURF_GRIDS: bool = Field(False, description="Save surface temperature grids")
    # SAVE_ALB_GRIDS: bool = Field(False, description="Save albedo grids")
    #
    # SAVE_QSW_PIXELS: bool = Field(False, description="Save shortwave radiation pixels")
    # SAVE_QLW_PIXELS: bool = Field(False, description="Save longwave radiation pixels")
    # SAVE_TSURF_PIXELS: bool = Field(False, description="Save surface temperature pixels")
    # SAVE_ALB_PIXELS: bool = Field(False, description="Save albedo pixels")

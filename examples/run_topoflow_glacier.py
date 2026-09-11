import argparse

import numpy as np
import pandas as pd
from pyprojroot import here

from topoflow_glacier import BmiTopoflowGlacier, __version__, configure_logging, logger


def run_topoflow_glacier(make_plot: bool) -> None:
    """The main function for running topoflow glacier

    Parameters
    ----------
    make_plot: bool
        Saves the output file as a plot
    """
    configure_logging()
    logger.info(f"Running Topoflow-Glacier version: {__version__}")

    bmi_cfg_file = here() / "config/cat-3062920.yaml"

    logger.debug("Creating an instance of an BMI_LSTM model object")
    model = BmiTopoflowGlacier()

    logger.debug("Initializing the BMI")
    model.initialize(bmi_cfg_file)

    logger.debug("Gathering input data")
    _df = pd.read_csv(here() / model.cfg.forcing_file)
    _df["Time"] = pd.to_datetime(_df["Time"])

    start_datetime = pd.to_datetime(model.cfg.start_time, format="%Y%m%d%H")
    end_datetime = pd.to_datetime(model.cfg.end_time, format="%Y%m%d%H")

    logger.info(f"Filtering forcings from {start_datetime} to {end_datetime}")

    df = _df[(_df["Time"] >= start_datetime) & (_df["Time"] <= end_datetime)].copy()

    logger.debug("Loop through the inputs, set the forcing values, and update the model...")
    precip_data = df["RAINRATE"].values
    temp_data = df["T2D"].values
    long_wave_radiation = df["LWDOWN"].values
    short_wave_radiation = df["SWDOWN"].values
    air_pressure = df["PSFC"].values
    air_water_vapor = df["Q2D"].values  # specific humidity
    wind_speed = (
        ((df["U2D"]) ** 2 + (df["V2D"]) ** 2) ** 0.5
    ).values  # making one single wind speed based on U and V directions

    output_snow_melt = np.zeros(len(precip_data))
    output_ice_melt = np.zeros(len(precip_data))
    output_rh = np.zeros(len(precip_data))
    output_h_swe = np.zeros(len(precip_data))
    output_h_iwe = np.zeros(len(precip_data))
    output_h_snow = np.zeros(len(precip_data))
    output_h_ice = np.zeros(len(precip_data))
    output_m_total = np.zeros(len(precip_data))

    # dest_array = np.zeros(1)
    dest_array = np.zeros(1, dtype=np.float64)

    model.get_value("snowpack__depth", dest_array)
    logger.info(f"|- Starting Snow Height: {float(dest_array[0])}")

    model.get_value("glacier_ice__thickness", dest_array)
    logger.info(f"|- Starting Ice Height: {float(dest_array[0])}")

    inputs = model.get_input_var_names()
    logger.debug(f"BMI inputs: {inputs}")

    # logger.info(f"|- Starting Snow Height: {model.get_value('snowpack__depth', dest_array).item()}")
    # logger.info(f"|- Starting Ice Height: {model.get_value('glacier_ice__thickness', dest_array).item()}")

    for i in range(len(precip_data)):
        model.set_value(
            "atmosphere_water__liquid_equivalent_precipitation_rate", precip_data[i] * 10 ** (-3)
        )  # converting mm/hr to m/hr
        model.set_value("land_surface_air__temperature", model.K_to_C + temp_data[i])  # converting to Celcius
        model.set_value("land_surface_radiation~incoming~longwave__energy_flux", long_wave_radiation[i])
        model.set_value("land_surface_radiation~incoming~shortwave__energy_flux", short_wave_radiation[i])
        model.set_value("land_surface_air__pressure", air_pressure[i])
        model.set_value("atmosphere_air_water~vapor__relative_saturation", air_water_vapor[i])
        model.set_value("wind_speed_UV", wind_speed[i])

        model.update()

        # Similiar output saving to:
        # https://github.com/NGWPC/lstm/blob/341be45ed854442459ad2f4ed05532b5eb5406fe/lstm/run_lstm_bmi.py#L52C1-L54C27
        dest_array = np.zeros(1)
        model.get_value("atmosphere_bottom_air_water-vapor__relative_saturation", dest_array)
        output_rh[i : i + 1] = dest_array

        dest_array = np.zeros(1)
        model.get_value("snowpack__melt_volume_flux", dest_array)
        output_snow_melt[i : i + 1] = dest_array

        dest_array = np.zeros(1)
        model.get_value("glacier_ice__melt_volume_flux", dest_array)
        output_ice_melt[i : i + 1] = dest_array

        dest_array = np.zeros(1)
        model.get_value("snowpack__liquid-equivalent_depth", dest_array)
        output_h_swe[i : i + 1] = dest_array

        dest_array = np.zeros(1)
        model.get_value("glacier__liquid_equivalent_depth", dest_array)
        output_h_iwe[i : i + 1] = dest_array

        dest_array = np.zeros(1)
        model.get_value("snowpack__depth", dest_array)
        output_h_snow[i : i + 1] = dest_array

        dest_array = np.zeros(1)
        model.get_value("glacier_ice__thickness", dest_array)
        output_h_ice[i : i + 1] = dest_array

        dest_array = np.zeros(1)
        model.get_value("land_surface_water__runoff_volume_flux", dest_array)
        output_m_total[i : i + 1] = dest_array

    # Finalizing the BMI
    logger.debug("Finalizing the BMI...")
    model.finalize()

    output_m_total = output_m_total * model.da_m2  # converting m/sec melt to m3/sec

    logger.info(f"|- Final Timestep Relative Humitidy: {output_rh[-1]}")
    logger.info(f"|- Final Timestep Snow Melt: {output_snow_melt[-1]}")
    logger.info(f"|- Final Timestep Ice Melt: {output_ice_melt[-1]}")
    logger.info(f"|- Final Timestep Height SWE: {output_h_swe[-1]}")
    logger.info(f"|- Final Timestep Height IWE: {output_h_iwe[-1]}")
    logger.info(f"|- Final Timestep Snow Height: {output_h_snow[-1]}")
    logger.info(f"|- Final Timestep Ice Height: {output_h_ice[-1]}")
    logger.info(f"|- Final Timestep Runoff from melt: {output_m_total[-1]}")

    # NOTE: To compare with the original TOPOFLOW we're using a convolution approach to mock routing
    # The original topoflow has a routing module included for flow
    # This will not ship with the code, but be used for a benchmark
    weights = np.zeros(20) + 0.05
    output_m_total = np.convolve(output_m_total, weights, mode="full")
    output_m_total = output_m_total[: len(output_h_ice)]
    if make_plot:
        from datetime import timedelta

        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        # Use the actual timestamps from the filtered DataFrame instead of generating them
        time_series = df["Time"].tolist()

        # Create the first figure with 2 vertically stacked subplots
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Top subplot: Snow Height (line) and Snow Melt (bars)
        ax1_melt = ax1.twinx()  # Create second y-axis for snow melt

        # Snow height as line
        _ = ax1.plot(time_series, output_h_snow, "b-", linewidth=2, label="Snow Height")
        ax1.set_ylabel("Snow Height (m)", color="b")
        ax1.tick_params(axis="y")
        ax1.grid(True, alpha=0.3)

        # Snow melt as bars
        _ = ax1_melt.bar(
            time_series,
            output_snow_melt,
            width=timedelta(hours=12),
            color="grey",
            alpha=0.7,
            label="Snow Melt",
        )
        ax1_melt.set_ylabel("Snow Melt (m/sec)", color="grey")
        ax1_melt.tick_params(axis="y")

        ax1.set_title("Snow Height and Snow Melt")
        ax1.set_xlabel("Time")

        # Bottom subplot: Ice Height (line) and Ice Melt (bars)
        ax2_melt = ax2.twinx()  # Create second y-axis for ice melt

        # Ice height as line
        _ = ax2.plot(time_series, output_h_ice, "r-", linewidth=2, label="Ice Height")
        ax2.set_ylabel("Ice Height (m)", color="r")
        ax2.tick_params(axis="y")
        ax2.grid(True, alpha=0.3)

        # Ice melt as bars
        _ = ax2_melt.bar(
            time_series,
            output_ice_melt,
            width=timedelta(hours=12),
            color="orange",
            alpha=0.7,
            label="Ice Melt",
        )
        ax2_melt.set_ylabel("Ice Melt (m/sec)", color="orange")
        ax2_melt.tick_params(axis="y")

        ax2.set_title("Ice Height and Ice Melt")
        ax2.set_xlabel("Time")

        # Format x-axis labels for both subplots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # Every 10 days
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=2))  # Every 2 days for minor ticks
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add legends for both subplots
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1_melt, labels1_melt = ax1_melt.get_legend_handles_labels()
        ax1.legend(lines1 + lines1_melt, labels1 + labels1_melt, loc="upper right")

        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2_melt, labels2_melt = ax2_melt.get_legend_handles_labels()
        ax2.legend(lines2 + lines2_melt, labels2 + labels2_melt, loc="upper right")

        # Tight layout to prevent overlap
        plt.tight_layout()

        # Save the first figure
        output_file1 = here() / "examples/snow_ice_height_melt.png"
        output_file1.parent.mkdir(exist_ok=True)
        plt.savefig(output_file1, dpi=300, bbox_inches="tight")
        logger.info(f"2-panel snow/ice height and melt plot saved to: {output_file1}")

        _, ax_flow = plt.subplots(figsize=(12, 6))

        ax_precip = ax_flow.twinx()

        _ = ax_flow.plot(time_series, output_m_total, "r-", linewidth=2, label="Runoff")
        ax_flow.set_xlabel("Time")
        ax_flow.set_ylabel("Flow (m³/s)", color="r")
        ax_flow.tick_params(axis="y", labelcolor="r")
        ax_flow.grid(True, alpha=0.3)

        # Plot precipitation as bars (inverted, similar to your reference image)
        # Convert precip from mm/hr to mm for display
        _ = ax_precip.bar(
            time_series,
            precip_data,
            width=timedelta(hours=0.8),
            color="blue",
            alpha=0.7,
            label="Precipitation",
        )
        ax_precip.set_ylabel("Rainfall depth (mm)", color="b")
        ax_precip.tick_params(axis="y", labelcolor="b")

        # Invert the precipitation axis to show bars going downward from top
        ax_precip.invert_yaxis()

        # Format x-axis with better tick spacing and formatting
        ax_flow.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_flow.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Weekly ticks
        ax_flow.xaxis.set_minor_locator(mdates.DayLocator(interval=1))  # Daily minor ticks
        plt.setp(ax_flow.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Set title
        ax_flow.set_title("Topoflow-Glacier Cat-3062920 Hydrograph with Precipitation")

        # Add legend
        lines1, labels1 = ax_flow.get_legend_handles_labels()
        lines2, labels2 = ax_precip.get_legend_handles_labels()
        ax_flow.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Tight layout
        plt.tight_layout()

        # Save the second figure
        output_file2 = here() / "examples/runoff_hydrograph_with_precip.png"
        plt.savefig(output_file2, dpi=300, bbox_inches="tight")
        logger.info(f"Hydrograph with precipitation saved to: {output_file2}")

    logger.debug("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an example case for running topoflow-glacier at mount rainer using Hydrofabric v2.2"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable plotting functionality",
    )

    args = parser.parse_args()
    run_topoflow_glacier(args.plot)

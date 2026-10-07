import os
import time
from math import sqrt
from statistics import fmean
from typing import Dict, List, Optional

from utils.config import load_config
from utils.rng import RNG
from sensors.temp_sensor import TempSensor
from sensors.filters import hold_last, MovingAverageFilter
from controllers.onoff import OnOffThermostat
from controllers.predictive_onoff import PredictiveOnOff
from simulations.room_model import step_room
from simulations.environment import Environment
from plotting.plots import (
    plot_timeseries,
    plot_error,
    plot_duty,
    plot_predictive,
    plot_heater,
)


def run_scenario(
    scenario_path: str,
    seed: int | None = None,
    save_outputs: bool = True,
) -> dict[str, float]:
    scenario = load_config(scenario_path)

    if seed is not None:
        scenario.sim.seed = seed

    rng = RNG(scenario.sim.seed)

    env = Environment(
        base=scenario.env.base,
        amplitude=scenario.env.amplitude,
        period_s=scenario.env.period_s,
        door_drop_C=scenario.env.door_drop_C,
        door_start_s=scenario.env.door_start_s,
        door_duration_s=scenario.env.door_duration_s,
    )

    sensor = TempSensor(
        sigma=scenario.sensor.sigma,
        bias=scenario.sensor.bias,
        dropout_prob=scenario.sensor.dropout_prob,
        rng=rng,
    )

    # Choose the controller configured in the scenario.
    if getattr(scenario.controller, "type", "predictive_onoff") == "onoff":
        ctrl = OnOffThermostat(
            setpoint=scenario.controller.setpoint,
            deadband=scenario.controller.deadband,
            safety_high=scenario.controller.safety_high,
            state=0,
        )
        use_predictive = False
    else:
        ctrl = PredictiveOnOff(
            setpoint=scenario.controller.setpoint,
            deadband=scenario.controller.deadband,
            tau=scenario.controller.tau,
            safety_high=scenario.controller.safety_high,
            state=0,
        )
        use_predictive = True

    dt = scenario.sim.dt
    if dt <= 0:
        raise ValueError("Simulation dt must be greater than zero.")

    steps = int(scenario.sim.duration_s / dt)
    if steps <= 0:
        raise ValueError("Simulation duration must be at least one time step.")

    temperature = scenario.sim.init_T
    last_valid: Optional[float] = temperature

    keys = [
        "t", "T_true", "T_meas", "T_out", "setpoint", "heater",
        "error", "T_pred", "lower", "upper",
    ]
    log: Dict[str, List[float]] = {key: [] for key in keys}
    moving_average = MovingAverageFilter(window=5)

    for k in range(steps):
        t = k * dt
        outside_temperature = env.T_out(t)

        measurement = sensor.read(temperature)
        measurement = hold_last(measurement, last_valid)
        if measurement is None:
            measurement = temperature
        last_valid = measurement

        filtered = moving_average.update(measurement)

        if use_predictive:
            heater = ctrl.update(filtered, dt)
            predicted = ctrl.last_pred if ctrl.last_pred is not None else filtered
            lower = (
                ctrl.lower_threshold
                if ctrl.lower_threshold is not None
                else ctrl.setpoint - ctrl.deadband / 2
            )
            upper = (
                ctrl.upper_threshold
                if ctrl.upper_threshold is not None
                else ctrl.setpoint + ctrl.deadband / 2
            )
        else:
            heater = ctrl.update(filtered)
            predicted = filtered
            lower = ctrl.setpoint - ctrl.deadband / 2
            upper = ctrl.setpoint + ctrl.deadband / 2

        error = ctrl.setpoint - filtered

        log["t"].append(t)
        log["T_true"].append(temperature)
        log["T_meas"].append(filtered)
        log["T_out"].append(outside_temperature)
        log["setpoint"].append(ctrl.setpoint)
        log["heater"].append(heater)
        log["error"].append(error)
        log["T_pred"].append(predicted)
        log["lower"].append(lower)
        log["upper"].append(upper)

        temperature = step_room(
            temperature,
            heater,
            outside_temperature,
            scenario.model.R,
            scenario.model.C,
            scenario.model.P,
            dt,
            scenario.model.process_sigma,
            rng,
        )

    # Metrics are calculated for every run, whether or not files are saved.
    rmse = sqrt(fmean(error**2 for error in log["error"]))
    mean_abs_error = fmean(abs(error) for error in log["error"])
    duty_cycle = fmean(log["heater"])

    if save_outputs:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.splitext(os.path.basename(scenario_path))[0]
        log_dir = os.path.join("outputs", "logs")
        figure_dir = os.path.join("outputs", "figures")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)

        csv_path = os.path.join(log_dir, f"{base}-{timestamp}.csv")
        with open(csv_path, "w", newline="") as file:
            file.write(",".join(log.keys()) + "\n")
            for i in range(len(log["t"])):
                file.write(",".join(str(log[key][i]) for key in log) + "\n")

        plot_timeseries(log, os.path.join(figure_dir, f"{base}-temps-{timestamp}.png"))
        plot_heater(log, os.path.join(figure_dir, f"{base}-heater-{timestamp}.png"))
        plot_error(log, os.path.join(figure_dir, f"{base}-error-{timestamp}.png"))
        plot_duty(log, os.path.join(figure_dir, f"{base}-duty-{timestamp}.png"))

        if use_predictive:
            plot_predictive(
                log,
                os.path.join(figure_dir, f"{base}-predictive-{timestamp}.png"),
            )

        print(f"Wrote log to {csv_path}")
        print(f"Figures saved to {figure_dir}")

    return {
        "rmse_C": rmse,
        "mean_abs_error_C": mean_abs_error,
        "duty_cycle": duty_cycle,
    }

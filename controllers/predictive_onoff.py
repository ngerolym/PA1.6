from dataclasses import dataclass

@dataclass
class PredictiveOnOff:
    setpoint: float
    deadband: float
    tau: float = 10.0         # lookahead horizon (seconds)
    safety_high: float = 26.0
    state: int = 0            # 0=OFF, 1=ON

    # Exposed diagnostics for plotting/logging
    last_meas: float | None = None
    last_pred: float | None = None
    lower_threshold: float | None = None
    upper_threshold: float | None = None

    def update(self, meas_filtered: float, dt: float) -> int:
        lower = self.setpoint - self.deadband / 2.0
        upper = self.setpoint + self.deadband / 2.0

        # Safety cutoff takes priority over predictive control.
        if meas_filtered >= self.safety_high:
            self.state = 0
            predicted = meas_filtered
        else:
            # On the first reading or with invalid dt, use the current
            # measurement instead of estimating a trend.
            if self.last_meas is None or dt <= 0:
                predicted = meas_filtered
            else:
                rate = (meas_filtered - self.last_meas) / dt
                predicted = meas_filtered + self.tau * rate

            if predicted < lower:
                self.state = 1
            elif predicted > upper:
                self.state = 0
            # Inside the deadband, retain the previous state.
            self.last_meas = meas_filtered
            self.last_pred = predicted
            self.lower_threshold = lower
            self.upper_threshold = upper
            return self.state
        
        if meas_filtered >= self.safety_high:
            self.state = 0
            self._update_diag(meas_filtered, dt, meas_filtered)  # pred doesn't matter here
            return self.state

        # Predict a short time into the future using discrete derivative
        if self.last_meas is None:
            T_pred = meas_filtered
        else:
            dTdt = (meas_filtered - self.last_meas) / dt
            T_pred = meas_filtered + self.tau * dTdt

        lower = self.setpoint - self.deadband / 2.0
        upper = self.setpoint + self.deadband / 2.0

        # Decision using predicted temperature
        if T_pred < lower:
            self.state = 1
        elif T_pred > upper:
            self.state = 0
        # else: keep previous state

        self._update_diag(meas_filtered, dt, T_pred, lower, upper)
        return self.state

    def _update_diag(
        self,
        meas: float,
        dt: float,  # Kept to match existing calls; diagnostics don't use it.
        T_pred: float,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        """Store the latest measurement, prediction, and thresholds."""
        self.last_meas = meas
        self.last_pred = T_pred

        self.lower_threshold = (
            self.setpoint - self.deadband / 2.0 if lower is None else lower
        )
        self.upper_threshold = (
            self.setpoint + self.deadband / 2.0 if upper is None else upper
        )
        
        self.last_pred = T_pred
        self.last_meas = meas
        if lower is None or upper is None:
            lower = self.setpoint - self.deadband / 2.0
            upper = self.setpoint + self.deadband / 2.0
        self.lower_threshold = lower
        self.upper_threshold = upper

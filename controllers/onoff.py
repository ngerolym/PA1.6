from dataclasses import dataclass

@dataclass
class OnOffThermostat:
    setpoint: float
    deadband: float
    safety_high: float = 26.0
    state: int = 0  # 0=OFF, 1=ON

    def update(self, measured_temp: float) -> int:
        lower = self.setpoint - self.deadband / 2
        upper = self.setpoint + self.deadband / 2

        if measured_temp > self.safety_high:
            self.state = 0
        elif measured_temp < lower:
            self.state = 1
        elif measured_temp > upper:
            self.state = 0

        return self.state

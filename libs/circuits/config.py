import dataclasses


@dataclasses.dataclass
class SimulatorConfig:
  # Time (in milliseconds) of 1 step of the simulation.
  timestep: float = 10

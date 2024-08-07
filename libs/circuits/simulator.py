import functools
import typing as T

from circuits import config, generators, monitors, nodes, units


def _simulator_init(node: nodes.ModuleNode, cfg: config.SimulatorConfig):
  if isinstance(node, (generators.Generator, units.Unit, monitors.Monitor)):
    node.init(cfg)


def _simulator_reset(node: nodes.ModuleNode):
  if isinstance(node, (generators.Generator, units.Unit, monitors.Monitor)):
    node.reset()


def _simulator_step(node: nodes.ModuleNode):
  # Update values, no dependencies.
  if isinstance(node, (generators.Generator, units.Unit)):
    node.step()


def _simulator_populate(node: nodes.ModuleNode):
  # Read values, via dependencies.
  if isinstance(node, (units.Unit, monitors.Monitor)):
    node.populate()


class Simulator(nodes.ContainerNode):
  def __init__(self, **kwargs):
    super().__init__()
    self.cfg = config.SimulatorConfig(**kwargs)
    assert self.cfg.timestep > 0

    # Number of simulated steps.
    self.steps: int

    # Amount of simulated time (in miliseconds), which depends on the configured timestep.
    self.time: float

  def init(self):
    nodes.visit(self, functools.partial(_simulator_init, cfg=self.cfg))
    self.reset()

  def reset(self):
    nodes.visit(self, _simulator_reset)
    self.steps = 0
    self.time = 0.

  def step(self):
    nodes.visit(self, _simulator_step)
    nodes.visit(self, _simulator_populate)
    self.steps += 1
    self.time += self.cfg.timestep

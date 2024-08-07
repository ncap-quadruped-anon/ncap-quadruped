import typing as T

from circuits import nodes


class Group(nodes.ContainerNode):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

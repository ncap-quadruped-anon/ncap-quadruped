import itertools
import typing as T

import torch
from circuits import config
from utils import collections


class Node:
  # Generator for unique `Node` IDs that are deterministic based on `Node` instatiation order.
  _newid = itertools.count()

  def __init__(
    self,
    xy: tuple[int, int] = (0, 0),
    **kwargs,
  ):
    self.id: int = next(Node._newid)
    self.xy: tuple[int, int] = xy
    self.meta: collections.AttrDict = collections.AttrDict(kwargs)
    super().__init__()  # Required for multiple-inheritance MRO to `torch.nn.Module`.

  def init(self, cfg: config.SimulatorConfig):
    pass

  def reset(self):
    pass


class PrimitiveNode(Node, torch.nn.Module):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)


class ContainerNode(Node, torch.nn.ModuleDict):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)


ModuleNode = T.Union[PrimitiveNode, ContainerNode]


# TODO: Make visitor generator so can perform post-child logic.
# Preorder: dict(child_kwargs: Dict[], continue: boolean)
# Postorder: dict(child_kwargs: Dict[], continue: boolean)
def visit(
  root: ModuleNode,
  visitor: T.Callable[[ModuleNode], bool | None],
):
  def traverse(node: ModuleNode):
    traverse_subtree = visitor(node)
    if traverse_subtree == False or isinstance(node, PrimitiveNode): return
    for child in node.children():
      # Force unbroken subtree of ModuleNodes, only allowing regular Modules as leaves.
      if isinstance(child, (PrimitiveNode, ContainerNode)):
        traverse(child)

  traverse(root)

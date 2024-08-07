import pandas as pd
import torch


def summarize_parameters(
  module: torch.nn.Module,
  *,
  display: bool = True,
  results: bool = False,
  recurse: bool = True,
):
  data = []
  for name, param in module.named_parameters(recurse=recurse):
    data.append((name, tuple(param.size()), str(param.dtype), param.requires_grad, param.numel()))
  full = pd.DataFrame(
    data, columns=['name', 'size', 'dtype', 'requires_grad', 'count']
  ).set_index('name')
  summary = full[['requires_grad', 'count']].groupby(['requires_grad']).sum().reindex(
    index=[True, False], fill_value=0
  )
  if display:
    print(summary.to_string(line_width=300))
    print(full.to_string(line_width=300))
  return (summary, full) if results else None

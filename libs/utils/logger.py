import logging
import os
import pathlib
import typing as T

import numpy as np
import numpy.typing as npt
import pandas as pd
import utils.io
import yaml


class FileSaver:
  # log.file('dirname').save('filename', 'data')
  # log.file('dirname').save({'filename', 'data'})

  def __init__(self, dirpath: pathlib.Path):
    self.dirpath = dirpath

  @T.overload
  def save(self, first: str | pathlib.Path, second: T.Any):
    ...

  @T.overload
  def save(self, first: T.Mapping[str, T.Any], second: None = None):
    ...

  def save(self, first, second=None):
    if isinstance(first, (str, pathlib.Path)):
      # Store single file.
      utils.io.write_file(self.dirpath / first, second)
    elif isinstance(first, dict):
      # Store multiple files
      for filename, data in first.items():
        utils.io.write_file(self.dirpath / filename, data)
    else:
      raise TypeError(f'Unrecognized argument type: {first}')
    return self


class KeyValueSaver:
  # log.keyval('main').store('key', 'value')
  # log.keyval('main').store({ 'key1': 'value1' })
  # log.keyval('main').append('key', 'value')
  # log.keyval('main').extend('key', ['value', 'value'])

  def __init__(self, filepath: pathlib.Path):
    self.filepath = filepath
    self._data: dict[str, T.Any] = {}

  @T.overload
  def store(self, first: str, second: T.Any, *, prefix: str = ''):
    ...

  @T.overload
  def store(self, first: T.Mapping[str, T.Any], second: None = None, *, prefix: str = ''):
    ...

  def store(self, first, second=None, *, prefix=''):
    if isinstance(first, str):
      # Store single key-value.
      self._data[f'{prefix}{first}'] = second
    elif isinstance(first, dict):
      # Store multiple keys-values.
      self._data.update({f'{prefix}{key}': value for key, value in first.items()})
    else:
      raise TypeError(f'Unrecognized argument type: {first}')
    return self

  def append(self, key: str, value: T.Any):
    if key in self._data:
      prev = self._data[key]
      if isinstance(prev, list):
        prev.append(value)
      else:
        raise RuntimeError(f'Appending value to non-list key: {key} <- {value}')
    else:
      self._data[key] = [value]
    return self

  def extend(self, key: str, values: T.Iterable[str | int | float]):
    if key in self._data:
      prev = self._data[key]
      if isinstance(prev, list):
        prev.extend(values)
      else:
        raise RuntimeError(f'Extending values to non-list key: {key} <- {list(values)}')
    else:
      self._data[key] = list(values)
    return self

  def save(self, **kwargs):
    kwargs.setdefault('sort_keys', False)
    with open(self.filepath, 'wt') as file:
      yaml.safe_dump(self._data, file, **kwargs)
    return self


TableSingleInput = str | int | float | None
TableSequenceInput = T.Iterable[str] | T.Iterable[int] | T.Iterable[float] | np.ndarray
TableSingleOrSequenceInput = TableSingleInput | TableSequenceInput
TableSingleValue = str | int | float | None
TableSequenceValue = list[str] | list[int] | list[float]
TableSingleOrSequenceValue = TableSingleValue | TableSequenceValue


class TableSaver:
  # log.table('train').store('column', 'value')
  # log.table('train').store({ 'column1': 'value1', 'column2': 'value2' })
  # log.table('train').append('column', 'value')
  # log.table('train').extend('column', ['value1', 'value2'])
  # log.table('train').row()

  def __init__(self, filepath: pathlib.Path):
    self.filepath = filepath
    self._active_row: dict[str, TableSingleOrSequenceValue] = {}
    self._unsaved_rows: list[dict[str, TableSingleValue]] = []
    self._saved_rows: list[dict[str, TableSingleValue]] = []
    self._columns: dict[str, T.Literal[True]] = {}  # Used as ordered set for quick lookup.
    self._overwrite: bool = True  # If columns change, need to overwrite file not just append.

  @T.overload
  def store(self, first: str, second: TableSingleOrSequenceInput, *, prefix: str = ''):
    ...

  @T.overload
  def store(
    self,
    first: T.Mapping[str, TableSingleOrSequenceInput],
    second: None = None,
    *,
    prefix: str = ''
  ):
    ...

  def store(self, first, second=None, *, prefix=''):
    if isinstance(first, str):
      # Store single value.
      items = {first: second}
    elif isinstance(first, dict):
      # Store multiple values.
      items = first
    else:
      raise TypeError(f'Unrecognized argument type: {first}')

    for column, value in items.items():
      column = f'{prefix}{column}'
      if isinstance(value, (int, float, str)) or value is None:
        self._active_row[column] = value
      elif isinstance(value, np.ndarray):
        self._active_row[column] = value.flatten().tolist()
      else:
        self._active_row[column] = list(value)
    return self

  def append(self, column: str, value: str | int | float):
    if column in self._active_row:
      prev = self._active_row[column]
      if isinstance(prev, list):
        prev.append(value)  # type: ignore
      else:
        raise RuntimeError(f'Appending value to non-list column: {column} <- {value}')
    else:
      self._active_row[column] = [value]  # type: ignore
    return self

  def extend(self, column: str, values: TableSequenceInput):
    lst = values.flatten().tolist() if isinstance(values, np.ndarray) else list(values)
    if column in self._active_row:
      prev = self._active_row[column]
      if isinstance(prev, list):
        prev.extend(lst)  # type: ignore
      else:
        raise RuntimeError(f'Extending values to non-list column: {column} <- {lst}')
    else:
      self._active_row[column] = lst  # type: ignore
    return self

  def row(self):
    row = {}
    for column, value in self._active_row.items():
      if isinstance(value, (str, int, float)) or value is None:
        row[column] = value
      elif isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], (int, float)):
          row[f'{column}/mean'] = np.mean(value).item()  # type: ignore
          row[f'{column}/std'] = np.std(value).item()  # type: ignore
          row[f'{column}/min'] = np.min(value).item()
          row[f'{column}/max'] = np.max(value).item()
          row[f'{column}/size'] = len(value)
        else:
          row[column] = '|'.join([str(x) for x in value])
    self._unsaved_rows.append(row)
    self._active_row = {}
    for column in row.keys():
      if column not in self._columns:
        self._columns[column] = True
        self._overwrite = True
    return self

  def save(self, **kwargs):
    kwargs.setdefault('index', False)
    if len(self._active_row) > 0: self.row()
    if self._overwrite:
      all_rows = self._saved_rows + self._unsaved_rows
      df = pd.DataFrame(all_rows, columns=list(self._columns.keys()))
      df.to_csv(self.filepath, **kwargs)
      self._saved_rows = all_rows
      self._unsaved_rows = []
      self._overwrite = False
    else:
      df = pd.DataFrame(self._unsaved_rows, columns=list(self._columns.keys()))
      df.to_csv(self.filepath, mode='a', header=False, **kwargs)
      self._saved_rows.extend(self._unsaved_rows)
      self._unsaved_rows = []
    return self


LOGGER_LEVELS = {
  'notset': logging.NOTSET,
  'debug': logging.DEBUG,
  'info': logging.INFO,
  'warning': logging.WARNING,
  'error': logging.ERROR,
  'critical': logging.CRITICAL,
}


class Logger:
  def __init__(
    self,
    log_dir: os.PathLike | str = './logs',
    *,
    name: str = '',
    level: T.Literal['notset', 'debug', 'info', 'warning', 'error', 'critical'] = 'notset',
  ):
    self.log_dir = pathlib.Path(log_dir).absolute()
    self.name = name or __name__
    self.level = level
    self._logger = logging.getLogger(self.name)
    self._logger.setLevel(LOGGER_LEVELS[level])

    # Initialize state for incremental savers.
    self._keyvals: dict[str, KeyValueSaver] = {}
    self._tables: dict[str, TableSaver] = {}
    self._active_keyval: KeyValueSaver | None = None
    self._active_table: TableSaver | None = None

  def __del__(self):
    self.finalize()

  def initialize(self):
    pass

  def finalize(self):
    self.save_all()

  # ================================================================================================
  # File logging (atomic)

  def file(self, *subdirpaths: str | os.PathLike):
    full_dirpath = pathlib.Path(self.log_dir, *subdirpaths)
    full_dirpath.mkdir(parents=True, exist_ok=True)
    return FileSaver(full_dirpath)

  @property
  def fi(self):
    return self.file()

  # TODO: Add image (img), video (vid), plot (plt).

  # ================================================================================================
  # File logging (incremental)

  def keyval(self, filepath: str = ''):
    # Empty filepath returns the active saver or creates a default saver.
    if not filepath and self._active_keyval is not None:
      return self._active_keyval
    filepath = filepath or 'keyval.yaml'

    # Non-empty filepath returns the existing saver or creates a new saver.
    if filepath in self._keyvals:
      self._active_keyval = self._keyvals[filepath]
    else:
      full_filepath = self.log_dir / filepath
      full_filepath.parent.mkdir(parents=True, exist_ok=True)
      saver = KeyValueSaver(full_filepath)
      self._keyvals[filepath] = saver
      self._active_keyval = saver
    return self._active_keyval

  @property
  def kv(self):
    return self.keyval()

  def table(self, filepath: str = ''):
    # Empty filepath returns the active saver or creates a default saver.
    if not filepath and self._active_table is not None:
      return self._active_table
    filepath = filepath or 'table.csv'

    # Non-empty filepath returns the existing saver or creates a new saver.
    if filepath in self._tables:
      self._active_table = self._tables[filepath]
    else:
      full_filepath = self.log_dir / filepath
      full_filepath.parent.mkdir(parents=True, exist_ok=True)
      saver = TableSaver(full_filepath)
      self._tables[filepath] = saver
      self._active_table = saver
    return self._active_table

  @property
  def tab(self):
    return self.table()

  def save_all(self):
    for keyval in self._keyvals.values():
      keyval.save()
    for table in self._tables.values():
      table.save()

  # ================================================================================================
  # Message logging

  def debug(self, msg: str, *args, **kwargs):
    self._logger.debug(msg, *args, **kwargs)

  def info(self, msg: str, *args, **kwargs):
    self._logger.info(msg, *args, **kwargs)

  def warning(self, msg: str, *args, **kwargs):
    self._logger.warning(msg, *args, **kwargs)

  def error(self, msg: str, *args, **kwargs):
    self._logger.error(msg, *args, **kwargs)

  def critical(self, msg: str, *args, **kwargs):
    self._logger.critical(msg, *args, **kwargs)

  def exception(self, msg: str, *args, **kwargs):
    self._logger.exception(msg, *args, **kwargs)

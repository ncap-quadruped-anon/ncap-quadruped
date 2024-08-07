import os

from hydra.core import config_search_path, plugins
from hydra.plugins import search_path_plugin

from .train import *
from .play import *
from .utils import *

# Extend Hydra config search path to include infrastructure configs. See
# https://github.com/facebookresearch/hydra/tree/main/examples/plugins/example_searchpath_plugin.
TONIC_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), 'configs')
class TonicSearchPathPlugin(search_path_plugin.SearchPathPlugin):
    def manipulate_search_path(self, search_path: config_search_path.ConfigSearchPath) -> None:
        search_path.append(provider='tonic', path=f'file://{TONIC_CONFIGS_PATH}')


plugins.Plugins.instance().register(TonicSearchPathPlugin)
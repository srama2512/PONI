from typing import List, Optional, Union

from habitat.config.default import _C, CN, CONFIG_FILE_SEPARATOR

from . import hlab_sensors  # Needed to register sensors


_C = _C.clone()

# -----------------------------------------------------------------------------
# SEMANTIC OBJECT SENSOR
# -----------------------------------------------------------------------------
_C.TASK.SEMANTIC_CATEGORY_SENSOR = CN()
_C.TASK.SEMANTIC_CATEGORY_SENSOR.TYPE = "SemanticCategorySensor"
_C.TASK.SEMANTIC_CATEGORY_SENSOR.HEIGHT = 480
_C.TASK.SEMANTIC_CATEGORY_SENSOR.WIDTH = 640
_C.TASK.SEMANTIC_CATEGORY_SENSOR.DATASET = "mp3d"


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config

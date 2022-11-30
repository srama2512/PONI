from typing import Any

import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator


MP3D_CATEGORY_MAPPING = {
    "chair": 0,
    "table": 1,
    "picture": 2,
    "cabinet": 3,
    "cushion": 4,
    "sofa": 5,
    "bed": 6,
    "chest_of_drawers": 7,
    "plant": 8,
    "sink": 9,
    "toilet": 10,
    "stool": 11,
    "towel": 12,
    "tv_monitor": 13,
    "shower": 14,
    "bathtub": 15,
    "counter": 16,
    "fireplace": 17,
    "gym_equipment": 18,
    "seating": 19,
    "clothes": 20,
}

GIBSON_CATEGORY_MAPPING = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
}


@registry.register_sensor(name="SemanticCategorySensor")
class SemanticCategorySensor(Sensor):
    r"""Lists the object categories for each pixel location.
    Args:
        sim: reference to the simulator for calculating task observations.
    """
    cls_uuid: str = "semantic_category"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._current_episode_id = None
        self.mapping = None
        self._initialize_category_mappings(config)

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _initialize_category_mappings(self, config):
        if config.DATASET == "mp3d":
            self.category_to_task_category_id = MP3D_CATEGORY_MAPPING
        else:
            self.category_to_task_category_id = GIBSON_CATEGORY_MAPPING
        self.num_task_categories = (
            np.max(list(self.category_to_task_category_id.values())) + 1
        )

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            shape=(self.config.HEIGHT, self.config.WIDTH),
            dtype=np.int32,
        )

    def get_observation(self, *args: Any, observations, episode, **kwargs: Any):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if self._current_episode_id != episode_uniq_id:
            self._current_episode_id = episode_uniq_id
            # Get mapping from instance id to task id
            scene = self._sim.semantic_annotations()
            self.instance_id_to_task_id = (
                np.ones((len(scene.objects),), dtype=np.int64) * -1
            )
            for obj in scene.objects:
                if obj is None:
                    continue
                obj_inst_id = int(obj.id.split("_")[-1])
                obj_name = obj.category.name()
                if obj_name in self.category_to_task_category_id.keys():
                    obj_task_id = self.category_to_task_category_id[obj_name]
                else:
                    obj_task_id = self.num_task_categories
                self.instance_id_to_task_id[obj_inst_id] = obj_task_id

        # Pre-process semantic observations to remove invalid values
        semantic = np.copy(observations["semantic"])
        semantic[semantic >= self.instance_id_to_task_id.shape[0]] = 0
        # Map from instance id to task id
        semantic_object = np.take(self.instance_id_to_task_id, semantic)
        semantic_object = semantic_object.astype(np.int32)

        return semantic_object

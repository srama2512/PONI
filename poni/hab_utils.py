import collections

import habitat_sim
import numpy as np

import trimesh
from habitat.utils.visualizations import maps

from sklearn.cluster import DBSCAN


def dense_sampling_trimesh(triangles, density=100.0, max_points=200000):
    # Create trimesh mesh from triangles
    t_vertices = triangles.reshape(-1, 3)
    t_faces = np.arange(0, t_vertices.shape[0]).reshape(-1, 3)
    t_mesh = trimesh.Trimesh(vertices=t_vertices, faces=t_faces)
    surface_area = t_mesh.area
    n_points = min(int(surface_area * density), max_points)
    t_pts, _ = trimesh.sample.sample_surface_even(t_mesh, n_points)
    return t_pts


def make_configuration(scene_path, scene_dataset_config=None, radius=0.18, height=0.88):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    if scene_dataset_config is not None:
        backend_cfg.scene_dataset_config_file = scene_dataset_config

    # agent configuration
    ## depth config
    depth_sensor_cfg = habitat_sim.CameraSensorSpec()
    depth_sensor_cfg.uuid = "depth"
    depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_cfg.resolution = [1080, 960]
    ## semantic config
    semantic_sensor_cfg = habitat_sim.CameraSensorSpec()
    semantic_sensor_cfg.uuid = "semantic"
    semantic_sensor_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_cfg.resolution = [1080, 960]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = height
    agent_cfg.radius = radius
    agent_cfg.sensor_specifications = [depth_sensor_cfg, semantic_sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_sim(glb_path):
    cfg = make_configuration(glb_path)
    sim = habitat_sim.Simulator(cfg)
    if not sim.pathfinder.is_loaded:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    return sim


def get_dense_navigable_points(sim, sampling_resolution=0.05):

    navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
    navmesh_vertices = navmesh_vertices.reshape(-1, 3, 3)
    navigable_points = []
    for face in navmesh_vertices:
        p1 = face[0]
        p2 = face[1]
        p3 = face[2]

        navigable_points.append(p1)
        navigable_points.append(p2)
        navigable_points.append(p3)

        ps = dense_sampling_util(p1, p2, p3, sampling_resolution)
        navigable_points += ps

    return navigable_points


def dense_sampling_util(p1, p2, p3, sampling_resolution):
    n1 = p2 - p1
    d1 = np.linalg.norm(n1)
    n1 = n1 / d1
    n2 = p3 - p1
    d2 = np.linalg.norm(n2)
    n2 = n2 / d2

    dense_points = [p1, p2, p3]
    for i in np.arange(0, d1, sampling_resolution):
        b = (d1 - i) * d2 / d1

        js = np.array(
            [[1, i, j] for j in np.arange(0, b, sampling_resolution)]
        )  # (N, 3)
        if len(js) == 0:
            continue
        x = np.stack([p1, n1, n2], axis=1)  # (D, 3)
        ps = np.matmul(js, x.T)
        for p in ps:
            dense_points.append(p)
    return dense_points


def get_floor_heights(sim, sampling_resolution=0.10):
    """Get heights of different floors in a scene. This is done in two steps.
    (1) Randomly samples navigable points in the scene.
    (2) Cluster the points based on discretized y coordinates to get floors.

    Args:
        sim - habitat simulator instance
        max_points_to_sample - number of navigable points to randomly sample
    """
    nav_points = get_dense_navigable_points(
        sim, sampling_resolution=sampling_resolution
    )
    nav_points = np.stack(nav_points, axis=0)
    y_coors = np.around(nav_points[:, 1], decimals=1)
    # Remove outliers (like staircases)
    y_counter = collections.Counter(y_coors)
    y_counts = y_counter.most_common(None)
    for y, count in y_counts:
        if count < int(len(nav_points) * 0.005):
            y_coors = y_coors[y_coors != y]
    # cluster Y coordinates
    min_samples = int(0.10 * len(y_coors))
    clustering = DBSCAN(eps=0.75, min_samples=min_samples).fit(y_coors[:, np.newaxis])
    c_labels = clustering.labels_
    n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
    # get floor extents in Y
    # each cluster corresponds to points from 1 floor
    floor_extents = []
    core_sample_y = y_coors[clustering.core_sample_indices_]
    core_sample_labels = c_labels[clustering.core_sample_indices_]
    for i in range(n_clusters):
        floor_min = core_sample_y[core_sample_labels == i].min().item()
        floor_max = core_sample_y[core_sample_labels == i].max().item()
        floor_mean = core_sample_y[core_sample_labels == i].mean().item()
        floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})
    floor_extents = sorted(floor_extents, key=lambda x: x["mean"])

    # reject floors that have too few points
    max_points = 0
    for fext in floor_extents:
        top_down_map = maps.get_topdown_map(sim.pathfinder, fext["mean"])
        max_points = max(np.count_nonzero(top_down_map), max_points)
    clean_floor_extents = []
    for fext in floor_extents:
        top_down_map = maps.get_topdown_map(sim.pathfinder, fext["mean"])
        num_points = np.count_nonzero(top_down_map)
        if num_points < 0.2 * max_points:
            continue
        clean_floor_extents.append(fext)

    return clean_floor_extents


def get_navmesh_extents_at_y(sim, y_bounds=None):
    if y_bounds is None:
        lower_bound, upper_bound = sim.pathfinder.get_bounds()
    else:
        assert len(y_bounds) == 2
        assert y_bounds[0] < y_bounds[1]
        navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
        navmesh_vertices = navmesh_vertices[
            (y_bounds[0] <= navmesh_vertices[:, 1])
            & (navmesh_vertices[:, 1] <= y_bounds[1])
        ]
        lower_bound = navmesh_vertices.min(axis=0)
        upper_bound = navmesh_vertices.max(axis=0)
    return lower_bound, upper_bound

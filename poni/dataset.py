import gc
import cv2
import bz2
import math
import json
import tqdm
import h5py
import glob
import torch
import random
import numpy as np
import os.path as osp
import _pickle as cPickle
import skimage.morphology as skmp

from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import Dataset
from poni.geometry import (
    spatial_transform_map,
    crop_map,
    get_frontiers_np,
)
from poni.constants import (
    SPLIT_SCENES,
    OBJECT_CATEGORIES,
    INV_OBJECT_CATEGORY_MAP,
    NUM_OBJECT_CATEGORIES,
    # General constants
    CAT_OFFSET,
    FLOOR_ID,
    # Coloring
    d3_40_colors_rgb,
    gibson_palette,
)
from poni.fmm_planner import FMMPlanner
from einops import asnumpy, repeat
from matplotlib import font_manager

MIN_OBJECTS_THRESH = 4
EPS = 1e-10


def is_int(s):
    try:
        int(s)
        return True
    except:
        return False


class SemanticMapDataset(Dataset):
    grid_size = 0.05 # m
    object_boundary = 1.0 # m
    def __init__(
        self,
        cfg,
        split='train',
        scf_name=None,
        seed=None,
    ):
        self.cfg = cfg
        self.dset = cfg.dset_name
        # Seed the dataset
        if seed is None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
        else:
            random.seed(seed)
            np.random.seed(seed)
        # Load maps
        maps_path = sorted(glob.glob(osp.join(cfg.root, "*.h5")))
        # Load json info
        maps_info = json.load(open(osp.join(cfg.root, 'semmap_GT_info.json')))
        maps = {}
        names = []
        maps_xyz_info = {}
        for path in maps_path:
            scene_name = path.split('/')[-1].split('.')[0]
            if scene_name not in SPLIT_SCENES[self.dset][split]:
                continue
            if (scf_name is not None) and (scene_name not in scf_name):
                continue
            with h5py.File(path, 'r') as fp:
                floor_ids = sorted([key for key in fp.keys() if is_int(key)])
                for floor_id in floor_ids:
                    name = f'{scene_name}_{floor_id}'
                    if (scf_name is not None) and (name != scf_name):
                        continue
                    map_world_shift = maps_info[scene_name]['map_world_shift']
                    if floor_id not in maps_info[scene_name]:
                        continue
                    map_y = maps_info[scene_name][floor_id]['y_min']
                    resolution = maps_info[scene_name]['resolution']
                    map_semantic = np.array(fp[floor_id]['map_semantic'])
                    nuniq = len(np.unique(map_semantic))
                    if nuniq >= MIN_OBJECTS_THRESH + 2:
                        names.append(name)
                        maps[name] = self.convert_maps_to_oh(map_semantic)
                        maps_xyz_info[name] = {
                            'world_shift': map_world_shift,
                            'resolution': resolution,
                            'y': map_y,
                            'scene_name': scene_name,
                        }
        self.maps = maps
        self.names = sorted(names)
        self.maps_xyz_info = maps_xyz_info
        self.visibility_size = cfg.visibility_size
        # Pre-compute FMM dists for each semmap
        if self.cfg.fmm_dists_saved_root == '':
            self.fmm_dists = self.compute_fmm_dists()
        else:
            self.fmm_dists = {}
            for name in self.names:
                fname = f'{cfg.fmm_dists_saved_root}/{name}.pbz2'
                with bz2.BZ2File(fname, 'rb') as fp:
                    self.fmm_dists[name] = (cPickle.load(fp)).astype(np.float32)
        # Pre-compute navigable locations for each map
        self.nav_locs = self.compute_navigable_locations()

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        name = self.names[idx]
        semmap = self.maps[name]
        fmm_dists = self.fmm_dists[name]
        map_xyz_info = self.maps_xyz_info[name]
        nav_space = semmap[FLOOR_ID]
        nav_locs = self.nav_locs[name]
        # Create input and output maps
        if self.cfg.masking_mode == 'spath':
            spath = self.get_random_shortest_path(nav_space, nav_locs)
            input, label = self.create_spath_based_input_output_pairs(
                semmap, fmm_dists, spath, map_xyz_info,
            )
        else:
            raise ValueError(f"Masking mode {self.cfg.masking_mode} is not implemented!")
        return input, label

    def get_item_by_name(self, name):
        assert name in self.names
        idx = self.names.index(name)
        return self[idx]

    def convert_maps_to_oh(self, semmap):
        ncat = NUM_OBJECT_CATEGORIES[self.dset]
        semmap_oh = np.zeros((ncat, *semmap.shape), dtype=np.float32)
        for i in range(0, ncat):
            semmap_oh[i] = (semmap == i + CAT_OFFSET).astype(np.float32)
        return semmap_oh

    def plan_path(self, nav_space, start_loc, end_loc):
        planner = FMMPlanner(nav_space)
        planner.set_goal(end_loc)
        curr_loc = start_loc
        spath = [curr_loc]
        ctr = 0
        while True:
            ctr += 1
            if ctr > 10000:
                print("plan_path() --- Run into infinite loop!")
                break
            next_y, next_x, _, stop = planner.get_short_term_goal(curr_loc)
            if stop:
                break
            curr_loc = (next_y, next_x)
            spath.append(curr_loc)
        return spath

    def get_random_shortest_path(self, nav_space, nav_locs):
        planner = FMMPlanner(nav_space)
        ys, xs = nav_locs
        num_outer_trials = 0
        while True:
            num_outer_trials += 1
            if num_outer_trials > 1000:
                print(f"=======> Stuck in infinite outer loop in!")
                break
            # Pick a random start location
            rnd_ix = np.random.randint(0, xs.shape[0])
            start_x, start_y = xs[rnd_ix], ys[rnd_ix]
            planner.set_goal((start_y, start_x))
            # Ensure that this is reachable from other points in the scene
            rchble_mask = planner.fmm_dist < planner.fmm_dist.max().item()
            if np.count_nonzero(rchble_mask) < 20:
                continue
            rchble_y, rchble_x = np.where(rchble_mask)
            # Pick a random goal location
            rnd_ix = np.random.randint(0, rchble_x.shape[0])
            end_x, end_y = rchble_x[rnd_ix], rchble_y[rnd_ix]
            break
        # Plan path from start to goal
        spath = self.plan_path(nav_space, (start_y, start_x), (end_y, end_x))
        return spath

    def compute_fmm_dists(self):
        fmm_dists = {}
        selem = skmp.disk(int(self.object_boundary / self.grid_size))
        for name in tqdm.tqdm(self.names):
            semmap = self.maps[name]
            navmap = semmap[FLOOR_ID]
            dists = []
            for catmap in semmap:
                if np.count_nonzero(catmap) == 0:
                    fmm_dist = np.zeros(catmap.shape)
                    fmm_dist.fill(np.inf)
                else:
                    cat_navmap = skmp.binary_dilation(catmap, selem) != True
                    cat_navmap = 1 - cat_navmap
                    cat_navmap[navmap > 0] = 1
                    planner = FMMPlanner(cat_navmap)
                    planner.set_multi_goal(catmap)
                    fmm_dist = np.copy(planner.fmm_dist)
                dists.append(fmm_dist)
            fmm_dists[name] = np.stack(dists, axis=0).astype(np.float32)
        return fmm_dists

    def compute_object_pfs(self, fmm_dists):
        cutoff = self.cfg.object_pf_cutoff_dist
        opfs = torch.clamp((cutoff - fmm_dists) / cutoff, 0.0, 1.0)
        return opfs

    def compute_navigable_locations(self):
        nav_locs = {}
        for name in self.names:
            semmap = self.maps[name]
            navmap = semmap[FLOOR_ID]
            ys, xs = np.where(navmap)
            nav_locs[name] = (ys, xs)
        return nav_locs

    def get_world_coordinates(self, map_xy, world_xyz_info):
        shift_xyz = world_xyz_info['world_shift']
        resolution = world_xyz_info['resolution']
        world_y = world_xyz_info['y']
        world_xyz = (
            map_xy[0] * resolution + shift_xyz[0],
            world_y,
            map_xy[1] * resolution + shift_xyz[2],
        )
        return world_xyz

    def get_visibility_map(self, in_semmap, locations):
        """
        locations - list of [y, x] coordinates
        """
        vis_map = np.zeros(in_semmap.shape[1:], dtype=np.uint8)
        for i in range(len(locations)):
            y, x = locations[i]
            y, x = int(y), int(x)
            if self.cfg.masking_shape == 'square':
                S = int(self.visibility_size / self.grid_size / 2.0)
                vis_map[(y - S) : (y + S), (x - S) : (x + S)] = 1
            else:
                raise ValueError(f'Masking shape {self.cfg.masking_shape} not defined!')

        vis_map = torch.from_numpy(vis_map).float()
        return vis_map

    def create_spath_based_input_output_pairs(
        self, semmap, fmm_dists, spath, map_xyz_info
    ):
        out_semmap = torch.from_numpy(semmap)
        out_fmm_dists = torch.from_numpy(fmm_dists) * self.grid_size
        in_semmap = out_semmap.clone()
        vis_map = self.get_visibility_map(in_semmap, spath)
        in_semmap *= vis_map
        # Transform the maps about a random center and rotate by a random angle
        center = random.choice(spath)
        rot = random.uniform(-math.pi, math.pi)
        Wby2, Hby2 = out_semmap.shape[2] // 2, out_semmap.shape[1] // 2
        tform_trans = torch.Tensor([[center[1] - Wby2, center[0] - Hby2, 0]])
        tform_rot = torch.Tensor([[0, 0, rot]])
        (
            in_semmap, out_semmap, out_fmm_dists, agent_fmm_dist, out_masks
        ) = self.transform_input_output_pairs(
            in_semmap, out_semmap, out_fmm_dists, tform_trans, tform_rot)
        # Get real-world position and orientation of agent
        world_xyz = self.get_world_coordinates(center, map_xyz_info)
        world_heading = -rot # Agent turning leftward is positive in habitat
        scene_name = map_xyz_info['scene_name']
        object_pfs = self.compute_object_pfs(out_fmm_dists)
        return in_semmap, {
            'semmap': out_semmap,
            'fmm_dists': out_fmm_dists,
            'agent_fmm_dist': agent_fmm_dist,
            'object_pfs': object_pfs,
            'masks': out_masks,
            'world_xyz': world_xyz,
            'world_heading': world_heading,
            'scene_name': scene_name,
        }

    def transform_input_output_pairs(
        self, in_semmap, out_semmap, out_fmm_dists, tform_trans, tform_rot
    ):
        # Invert fmm_dists for transformations (since padding is zeros)
        max_dist = out_fmm_dists[out_fmm_dists != math.inf].max() + 1
        out_fmm_dists = 1 / (out_fmm_dists + EPS)
        # Expand to add batch dim
        in_semmap = in_semmap.unsqueeze(0)
        out_semmap = out_semmap.unsqueeze(0)
        out_fmm_dists = out_fmm_dists.unsqueeze(0)
        # Crop a large-enough map around agent
        _, N, H, W = in_semmap.shape
        crop_center = torch.Tensor([[W / 2.0, H / 2.0]]) + tform_trans[:, :2]
        map_size = int(2.0 * self.cfg.output_map_size / self.grid_size)
        in_semmap = crop_map(in_semmap, crop_center, map_size)
        out_semmap = crop_map(out_semmap, crop_center, map_size)
        out_fmm_dists = crop_map(out_fmm_dists, crop_center, map_size)
        # Rotate the map
        in_semmap = spatial_transform_map(in_semmap, tform_rot)
        out_semmap = spatial_transform_map(out_semmap, tform_rot)
        out_fmm_dists = spatial_transform_map(out_fmm_dists, tform_rot)
        # Crop out the appropriate size of the map
        _, N, H, W = in_semmap.shape
        map_center = torch.Tensor([[W / 2.0, H / 2.0]])
        map_size = int(self.cfg.output_map_size / self.grid_size)
        in_semmap = crop_map(in_semmap, map_center, map_size, 'nearest')
        out_semmap = crop_map(out_semmap, map_center, map_size, 'nearest')
        out_fmm_dists = crop_map(out_fmm_dists, map_center, map_size, 'nearest')
        # Create a scaling-mask for the loss function. By default, select
        # only navigable / object regions (where fmm_dists exists).
        out_masks = (out_semmap[0, FLOOR_ID] >= 0.5).float() # (H, W)
        out_masks = repeat(out_masks, 'h w -> () n h w', n=N)
        # Mask out potential fields based on input regions
        if self.cfg.potential_function_masking:
            # Compute frontier locations
            unk_map = (
                torch.max(in_semmap, dim=1).values[0] < 0.5
            ).float().numpy() # (H, W)
            free_map = (in_semmap[0, FLOOR_ID] >= 0.5).float().numpy() # (H, W)
            frontiers = get_frontiers_np(unk_map, free_map)
            frontiers = torch.from_numpy(frontiers).float().unsqueeze(0).unsqueeze(1)

            # Dilate the frontiers mask
            frontiers_mask = torch.nn.functional.max_pool2d(frontiers, 7, stride=1, padding=3)
            # Scaling loss at the frontiers
            alpha = self.cfg.potential_function_frontier_scaling
            # Scaling loss at the non-visible regions
            beta = self.cfg.potential_function_non_visible_scaling
            visibility_mask = (in_semmap.sum(dim=1, keepdim=True) > 0).float()
            # Scaling loss at the visible & non-frontier regions
            gamma = self.cfg.potential_function_non_frontier_scaling
            not_frontier_or_visible = (1 - visibility_mask) * (1 - frontiers_mask)
            visible_and_not_frontier = visibility_mask * (1 - frontiers_mask)
            # Compute final mask
            out_masks = out_masks *(
                visible_and_not_frontier * gamma + \
                not_frontier_or_visible * beta + \
                frontiers_mask * alpha
            )
        # Remove batch dim
        in_semmap = in_semmap.squeeze(0)
        out_semmap = out_semmap.squeeze(0)
        out_fmm_dists = out_fmm_dists.squeeze(0)
        out_masks = out_masks.squeeze(0)
        # Invert fmm_dists for transformations (since padding, new pixels, etc are zeros)
        out_fmm_dists = torch.clamp(1 / (out_fmm_dists + EPS), 0.0, max_dist)
        # Compute distance from agent to all locations on the map
        nav_map = out_semmap[FLOOR_ID].numpy() # (H, W)
        planner = FMMPlanner(nav_map)
        agent_map = np.zeros(nav_map.shape, dtype=np.float32)
        Hby2, Wby2 = agent_map.shape[0] // 2, agent_map.shape[1] // 2
        agent_map[Hby2 - 1 : Hby2 + 2, Wby2 - 1 : Wby2 + 2] = 1
        selem = skmp.disk(int(self.object_boundary / 2.0 / self.grid_size))
        agent_map = skmp.binary_dilation(agent_map, selem) != True
        agent_map = 1 - agent_map
        planner.set_multi_goal(agent_map)
        agent_fmm_dist = torch.from_numpy(planner.fmm_dist) * self.grid_size

        return in_semmap, out_semmap, out_fmm_dists, agent_fmm_dist, out_masks

    @staticmethod
    def visualize_map(semmap, bg=1.0, dataset='gibson'):
        n_cat = semmap.shape[0] - 2 # Exclude floor and wall
        def compress_semmap(semmap):
            c_map = np.zeros((semmap.shape[1], semmap.shape[2]))
            for i in range(semmap.shape[0]):
                c_map[semmap[i] > 0.] = i+1
            return c_map

        palette = [
            int(bg * 255), int(bg * 255), int(bg * 255), # Out of bounds
            230, 230, 230, # Free space
            77, 77, 77, # Obstacles
        ]
        if dataset == 'gibson':
            palette += [int(x * 255.) for x in gibson_palette[15:]]
        else:
            palette += [c for color in d3_40_colors_rgb[:n_cat]
                        for c in color.tolist()]
        semmap = asnumpy(semmap)
        c_map = compress_semmap(semmap)
        semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
        semantic_img.putpalette(palette)
        semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = np.array(semantic_img)

        return semantic_img

    @staticmethod
    def visualize_object_pfs(
        in_semmap, semmap, object_pfs, dirs=None, locs=None, dataset='gibson'
    ):
        """
        semmap - (C, H, W)
        object_pfs - (C, H, W)
        """
        in_semmap = asnumpy(in_semmap)
        semmap = asnumpy(semmap)
        semmap_rgb = SemanticMapDataset.visualize_map(in_semmap, bg=1.0, dataset=dataset)
        red_image = np.zeros_like(semmap_rgb)
        red_image[..., 0] = 255
        object_pfs = asnumpy(object_pfs)
        vis_ims = []
        for i in range(0, semmap.shape[0]):
            opf = object_pfs[i][..., np.newaxis]
            sm = np.copy(semmap_rgb)
            smpf = red_image * opf + sm * (1 - opf)
            smpf = smpf.astype(np.uint8)
            # Highlight directions
            if dirs is not None and dirs[i] is not None:
                dir = math.radians(dirs[i])
                sx, sy = sm.shape[1] // 2, sm.shape[0] // 2
                ex = int(sx + 200 * math.cos(dir))
                ey = int(sy + 200 * math.sin(dir))
                cv2.line(smpf, (sx, sy), (ex, ey), (0, 255, 0), 3)
            # Highlight object locations
            smpf[semmap[i] > 0, :] = np.array([0, 0, 255])
            # Highlight location targets
            if locs is not None:
                H, W = semmap.shape[1:]
                x, y = locs[i]
                if x >= 0 and y >= 0:
                    x, y = int(x * W), (y * H)
                    cv2.circle(smpf, (x, y), 3, (0, 255, 0), -1)
            vis_ims.append(smpf)

        return vis_ims

    @staticmethod
    def visualize_object_category_pf(semmap, object_pfs, cat_id, dset):
        """
        semmap - (C, H, W)
        object_pfs - (C, H, W)
        cat_id - integer
        """
        semmap = asnumpy(semmap)
        offset = OBJECT_CATEGORIES[dset].index('chair')
        object_pfs = asnumpy(object_pfs)[cat_id + offset] # (H, W)
        object_pfs = object_pfs[..., np.newaxis] # (H, W)
        semmap_rgb = SemanticMapDataset.visualize_map(semmap, bg=1.0, dataset=dset)
        red_image = np.zeros_like(semmap_rgb)
        red_image[..., 0] = 255
        smpf = red_image * object_pfs + semmap_rgb * (1 - object_pfs)
        smpf = smpf.astype(np.uint8)

        return smpf

    def visualize_area_pf(semmap, area_pfs, dset='gibson'):
        """
        semmap - (C, H, W)
        are_pfs - (1, H, W)
        """
        semmap = asnumpy(semmap)
        pfs = asnumpy(area_pfs)[0] # (H, W)
        pfs = pfs[..., np.newaxis] # (H, W)
        semmap_rgb = SemanticMapDataset.visualize_map(semmap, bg=1.0, dataset=dset)
        red_image = np.zeros_like(semmap_rgb)
        red_image[..., 0] = 255
        smpf = red_image * pfs + semmap_rgb * (1 - pfs)
        smpf = smpf.astype(np.uint8)
        
        return smpf

    @staticmethod
    def combine_image_grid(
        in_semmap, out_semmap, gt_object_pfs, pred_object_pfs=None,
        gt_acts=None, gt_area_pfs=None, pred_area_pfs=None, dset=None,
        n_per_row=8, pad=2, border_color=200, output_width=1024,
    ):
        img_and_titles = [
            (in_semmap, 'Input map'), (out_semmap, 'Full output map')
        ]
        if gt_area_pfs is not None:
            img_and_titles.append((gt_area_pfs, 'GT Area map'))
        if pred_area_pfs is not None:
            img_and_titles.append((pred_area_pfs, 'Pred Area map'))
        for i, cat in INV_OBJECT_CATEGORY_MAP[dset].items():
            acts_suffix = ''
            if gt_acts is not None:
                acts_suffix = f'(act: {gt_acts[i].item():d})'
            if cat in ['wall', 'floor']:
                continue
            if pred_object_pfs is None:
                title = 'PF for ' + cat + acts_suffix
                img_and_titles.append((gt_object_pfs[i], title))
            else:
                title = 'GT PF for ' + cat + acts_suffix
                img_and_titles.append((gt_object_pfs[i], title))
                title = 'Pred PF for ' + cat + acts_suffix
                img_and_titles.append((pred_object_pfs[i], title))

        imgs = []
        for img, title in img_and_titles:
            cimg = SemanticMapDataset.add_title_to_image(img, title)
            # Pad image
            cimg = np.pad(cimg, ((pad, pad), (pad, pad), (0, 0)),
                          mode='constant', constant_values=border_color)
            imgs.append(cimg)

        # Convert to grid
        n_rows = len(imgs) // n_per_row
        if n_rows * n_per_row < len(imgs):
            n_rows += 1
        n_cols = min(len(imgs), n_per_row)
        H, W = imgs[0].shape[:2]
        grid_img = np.zeros((n_rows * H, n_cols * W, 3), dtype=np.uint8)
        for i, img in enumerate(imgs):
            r = i // n_per_row
            c = i % n_per_row
            grid_img[r * H : (r + 1) * H, c * W : (c + 1) * W] = img
        # Rescale image
        if output_width is not None:
            output_height = int(
                output_width * grid_img.shape[0] / grid_img.shape[1]
            )
            grid_img = cv2.resize(grid_img, (output_width, output_height))
        return grid_img

    @staticmethod
    def add_title_to_image(
        img: np.ndarray, title: str, font_size: int = 50, bg_color=200,
        fg_color=(0, 0, 255)
    ):
        font_img = np.zeros((font_size, img.shape[1], 3), dtype=np.uint8)
        font_img.fill(bg_color)
        font_img = Image.fromarray(font_img)
        draw = ImageDraw.Draw(font_img)
        # Find a font file
        mpl_font = font_manager.FontProperties(family="sans-serif", weight="bold")
        file = font_manager.findfont(mpl_font)
        font = ImageFont.truetype(font=file, size=25)
        draw.text((20, 5), title, fg_color, font=font)
        font_img = np.array(font_img)
        return np.concatenate([font_img, img], axis=0)


class SemanticMapPrecomputedDataset(SemanticMapDataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.dset = cfg.dset_name
        # Seed the dataset
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        # Load map paths
        map_paths = sorted(
            glob.glob(osp.join(cfg.root, split, f"**/*.pbz2"), recursive=True)
        )
        self.map_paths = map_paths
        # Both locations and directions cannot be enabled at the same time.
        assert not (self.cfg.enable_locations and self.cfg.enable_directions)

    def __len__(self):
        return len(self.map_paths)

    def compute_object_pfs(self, fmm_dists):
        cutoff = self.cfg.object_pf_cutoff_dist
        opfs = torch.clamp((cutoff - fmm_dists) / cutoff, 0.0, 1.0)
        return opfs

    def __getitem__(self, idx):
        with bz2.BZ2File(self.map_paths[idx], 'rb') as fp:
            data = cPickle.load(fp)
        # Convert cm -> m
        data['fmm_dists'] = data['fmm_dists'].astype(np.float32) / 100.0
        in_semmap = torch.from_numpy(data['in_semmap'])
        semmap = torch.from_numpy(data['semmap'])
        fmm_dists = torch.from_numpy(data['fmm_dists'])
        # Compute object_pfs
        object_pfs = self.compute_object_pfs(fmm_dists)
        loss_masks, masks, dirs, locs, area_pfs, acts, frontiers = self.get_masks_and_labels(
            in_semmap, semmap, fmm_dists
        )
        if self.cfg.potential_function_masking:
            object_pfs = torch.clamp(object_pfs * masks, 0.0, 1.0)

        input = {'semmap': in_semmap}
        ########################################################################
        # Optimizations for reducing memory usage during data-loading
        ########################################################################
        # Convert object_pfs to integers (0 -> 1000)
        object_pfs = (object_pfs * 1000.0).int()
        label = {
            'semmap': semmap,
            'object_pfs': object_pfs,
            'loss_masks': loss_masks,
        }
        # Convert area-pfs to integers (0 -> 1000)
        if area_pfs is not None:
            area_pfs = (area_pfs * 1000.0).int()
        ########################################################################

        if dirs is not None:
            label['dirs'] = dirs
        if locs is not None:
            label['locs'] = locs
        if area_pfs is not None:
            label['area_pfs'] = area_pfs
        if acts is not None:
            label['acts'] = acts
        if frontiers is not None:
            label['frontiers'] = frontiers
        # Free memory
        del data
        gc.collect()
        return input, label

    def get_masks_and_labels(self, in_semmap, out_semmap, out_fmm_dists):
        # Expand to add batch dim
        in_semmap = in_semmap.unsqueeze(0)
        out_semmap = out_semmap.unsqueeze(0)
        out_fmm_dists = out_fmm_dists.unsqueeze(0)
        N = in_semmap.shape[1]
        # Create a scaling-mask for the loss function / potential field
        # By default, select only navigable/object regions where fmm dist exists
        out_base_masks = torch.any(out_semmap, dim=1, keepdim=True) # (1, 1, H, W)
        out_base_masks = repeat(out_base_masks, '() () h w -> () n h w', n=N).float()
        ################### Build mask based on input regions ##################
        # Compute an advanced mask based on input regions.
        out_masks = torch.any(out_semmap, dim=1, keepdim=True) # (1, 1, H, W)
        out_masks = repeat(out_masks, '() () h w -> () n h w', n=N).float()
        # Compute frontier locations
        free_map = in_semmap[0, FLOOR_ID] # (H, W)
        # Dilate the free map
        if self.cfg.dilate_free_map:
            free_map = free_map.float().unsqueeze(0).unsqueeze(1)
            for i in range(self.cfg.dilate_iters):
                free_map = torch.nn.functional.max_pool2d(
                    free_map, 7, stride=1, padding=3
                )
            free_map = free_map.bool().squeeze(1).squeeze(0)
        exp_map = torch.any(in_semmap, dim=1)[0] # (H, W)
        exp_map = exp_map | free_map
        unk_map = ~exp_map
        unk_map = unk_map.numpy()
        free_map = free_map.numpy()
        frontiers = get_frontiers_np(unk_map, free_map) # (H, W)
        # Compute contours of frontiers
        contours = None
        if self.cfg.enable_unexp_area:
            contours, _ = cv2.findContours(
                frontiers.astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [contour[:, 0].tolist() for contour in contours] # Clean format
        frontiers = torch.from_numpy(frontiers).unsqueeze(0).unsqueeze(1)
        # Dilate the frontiers mask
        frontiers_mask = torch.nn.functional.max_pool2d(
            frontiers.float(), 7, stride=1, padding=3
        ).bool() # (1, N or 1, H, W)
        # Scaling at the frontiers
        alpha = self.cfg.potential_function_frontier_scaling
        # Scaling at the non-visible regions
        beta = self.cfg.potential_function_non_visible_scaling
        visibility_mask = torch.any(in_semmap, dim=1, keepdim=True) # (1, 1, H, W)
        # Scaling at the visible & non-frontier regions
        gamma = self.cfg.potential_function_non_frontier_scaling
        not_frontier_or_visible = ~(visibility_mask | frontiers_mask)
        visible_and_not_frontier = visibility_mask & (~frontiers_mask)
        # Compute final mask
        out_masks = out_masks * (
            visible_and_not_frontier * gamma + \
            not_frontier_or_visible * beta + \
            frontiers_mask * alpha
        )
        # Compute directions to each object from map center if needed
        ## For each category, pick the object nearest (euclidean distance)
        ## to the center.  Them compute the directions from center to
        ## this object. Conventions: East is 0, clockwise is positive
        out_dirs = None
        if self.cfg.enable_directions:
            out_dirs = []
            all_dirs = np.array(self.cfg.prediction_directions)
            ndir = len(self.cfg.prediction_directions)
            for sem_map in out_semmap[0]: # (H, W)
                sem_map = sem_map.cpu().numpy()
                H, W = sem_map.shape
                Hby2, Wby2 = H // 2, W // 2
                # Discover connected components (i.e., object instances)
                _, _, _, centroids = cv2.connectedComponentsWithStats(
                    sem_map.astype(np.uint8) * 255 , 4 , cv2.CV_32S
                )
                # Ignore 1st element of centroid since it's the image center
                centroids = centroids[1:]
                if len(centroids) == 0:
                    # class N is object missing class
                    out_dirs.append(ndir)
                    continue
                map_y, map_x = centroids[:, 1], centroids[:, 0]
                # Pick closest instance of the object
                dists = np.sqrt((map_y - Hby2) ** 2 + (map_x - Wby2) ** 2)
                min_idx = np.argmin(dists)
                obj_y, obj_x = map_y[min_idx], map_x[min_idx]
                obj_dir = np.arctan2(obj_y - Hby2, obj_x - Wby2)
                obj_dir = (np.rad2deg(obj_dir) + 360.0) % 360.0
                # Classify obj_dir into [0, ..., ndir-1] classes
                dir_cls = np.argmin(np.abs(all_dirs - obj_dir))
                out_dirs.append(dir_cls)
            out_dirs = torch.LongTensor(out_dirs).to(out_masks.device) # (N, )
        # Compute position to each object from map center if needed
        ## For each category, pick the object nearest (euclidean distance) to the center
        ## The compute the central position of the object in this map.
        ## Normalize the position b/w 0 to 1. Output is (x, y).
        ## Conventions: East is X, South is Y, map top-left is (0, 0)
        out_locs = None
        if self.cfg.enable_locations:
            out_locs = []
            for sem_map in out_semmap[0]: # (H, W)
                sem_map = sem_map.cpu().numpy()
                H, W = sem_map.shape
                Hby2, Wby2 = H // 2, W // 2
                # Discover connected components (i.e., object instances)
                _, _, _, centroids = cv2.connectedComponentsWithStats(
                    sem_map.astype(np.uint8) * 255 , 4 , cv2.CV_32S
                )
                # Ignore 1st element of centroid since it's the image center
                centroids = centroids[1:]
                if len(centroids) == 0:
                    out_locs.append((-1, -1))
                    continue
                map_y, map_x = centroids[:, 1], centroids[:, 0]
                # Pick closest instance of the object
                dists = np.sqrt((map_y - Hby2) ** 2 + (map_x - Wby2) ** 2)
                min_idx = np.argmin(dists)
                obj_y, obj_x = map_y[min_idx], map_x[min_idx]
                # Normalize this to (0, 1) range
                obj_y = obj_y / H
                obj_x = obj_x / W
                out_locs.append((obj_x, obj_y))
            out_locs = torch.Tensor(out_locs).to(out_masks.device) # (N, 2)
        # Compute action needed to reach each object from map center if needed.
        ## Assume that the agent is at the map center, facing right.
        out_acts = None
        if hasattr(self.cfg, 'enable_actions') and self.cfg.enable_actions:
            out_acts = []
            traversible = out_semmap[0, 0] | (~torch.any(out_semmap[0], dim=0)) # (H, W)
            planner = FMMPlanner(traversible.float().cpu().numpy())
            H, W = traversible.shape
            Hby2, Wby2 = H // 2, W // 2
            traversible[Hby2 - 3:Hby2 + 4, Wby2 - 3:Wby2 + 4] = 1
            for i, (sem_map, fmm_dist) in enumerate(zip(out_semmap[0], out_fmm_dists[0])): # (H, W)
                sem_map = sem_map.cpu().numpy()
                # Use pre-computed fmm dists
                fmm_dist = fmm_dist.cpu().numpy()
                H, W = sem_map.shape
                assert H == W
                map_resolution = self.cfg.output_map_size / H
                if not np.any(sem_map > 0):
                    out_acts.append(-1)
                    continue
                # planner.fmm_dist = np.floor(fmm_dist / map_resolution)
                goal_map = sem_map.astype(np.float32)
                selem = skmp.disk(0.5 / map_resolution)
                goal_map = skmp.binary_dilation(goal_map, selem) != True
                goal_map = 1 - goal_map * 1.
                planner.set_multi_goal(goal_map)

                start = (H // 2, W // 2)
                stg_x, stg_y, _, stop = planner.get_short_term_goal(start)
                if stop:
                    out_acts.append(0) # STOP
                else:
                    angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                            stg_y - start[1]))
                    angle_agent = 0.0
                    relative_angle = (angle_agent - angle_st_goal) % 360.0
                    if relative_angle > 180:
                        relative_angle -= 360
                    
                    if relative_angle > self.cfg.turn_angle / 2.0:
                        out_acts.append(3) # TURN-RIGHT
                    elif relative_angle < -self.cfg.turn_angle / 2.0:
                        out_acts.append(2) # TURN-RIGHT
                    else:
                        out_acts.append(1) # MOVE-FORWARD
            out_acts = torch.Tensor(out_acts).long().to(out_masks.device) # (N,)

        # Compute unexplored free-space starting from each frontier
        out_area_pfs = None
        if self.cfg.enable_unexp_area:
            floor_map = out_semmap[0, FLOOR_ID] # (H, W)
            unexp_map = ~torch.any(in_semmap[0], dim=0) # (H, W)
            unexp_floor_map = floor_map & unexp_map # (H, W)
            # Identify connected components of unexplored floor space
            unexp_floor_map = unexp_floor_map.cpu().numpy()
            unexp_floor_map = unexp_floor_map.astype(np.uint8) * 255
            ncomps, comp_labs, _, _ = cv2.connectedComponentsWithStats(
                unexp_floor_map, 4 , cv2.CV_32S
            )
            # Only select largest 5 contours
            largest_contours = sorted(
                contours, key=lambda cnt: len(cnt), reverse=True
            )[:5]
            contour_stats = [0.0 for _ in range(len(largest_contours))]
            # For each connected component, find the intersecting frontiers and
            # add area to them.
            kernel = np.ones((5, 5))
            for i in range(1, ncomps):
                comp = (comp_labs == i).astype(np.float32)
                comp_area = comp.sum().item() * (self.grid_size ** 2)
                # dilate
                comp = cv2.dilate(comp, kernel, iterations=1)
                # intersect with frontiers
                for j, contour in enumerate(largest_contours):
                    intersection = 0.0
                    for x, y in contour:
                        intersection += comp[y, x]
                    if intersection > 0:
                        contour_stats[j] += comp_area
            # Create out areas map
            out_area_pfs = torch.zeros_like(floor_map).float() # (H, W)
            if hasattr(self.cfg, 'normalize_area_by_constant'):
                normalize_area_by_constant = self.cfg.normalize_area_by_constant
            else:
                normalize_area_by_constant = False

            if normalize_area_by_constant:
                total_area = self.cfg.max_unexp_area
            else:
                total_area = floor_map.sum().item() * (self.grid_size ** 2) / 2.0
            for stat, contour in zip(contour_stats, largest_contours):
                # Use linear scoring
                score = np.clip(stat / (total_area + EPS), 0.0, 1.0)
                for x, y in contour:
                    out_area_pfs[y, x] = score
            # Dilate the area map
            out_area_pfs = out_area_pfs.unsqueeze(0).unsqueeze(1) # (1, 1, H, W)
            out_area_pfs = torch.nn.functional.max_pool2d(
                out_area_pfs, 7, stride=1, padding=3
            )
            out_area_pfs = out_area_pfs.squeeze(1) # (1, H, W)

        # Remove batch dim
        out_base_masks = out_base_masks.squeeze(0)
        out_masks = out_masks.squeeze(0)
        return out_base_masks, out_masks, out_dirs, out_locs, out_area_pfs, out_acts, contours
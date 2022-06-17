import os
import bz2
import tqdm
import argparse
import numpy as np
import _pickle as cPickle
import multiprocessing as mp
from poni.default import get_cfg
from poni.dataset import SemanticMapDataset

from einops import asnumpy


assert 'ACTIVE_DATASET' in os.environ
ACTIVE_DATASET = os.environ['ACTIVE_DATASET']
DATASET = ACTIVE_DATASET
OUTPUT_MAP_SIZE = 24.0
MASKING_MODE = "spath"
MASKING_SHAPE = "square"

SEED = 123
DATA_ROOT = "data/semantic_maps/{}/semantic_maps".format(DATASET)
FMM_DISTS_SAVED_ROOT = "data/semantic_maps/{}/fmm_dists_{}".format(
    DATASET, SEED
)
NUM_SAMPLES = {'train': 400000, 'val': 1000}
SAVE_ROOT = "data/semantic_maps/{}/precomputed_dataset_{}_{}_{}_{}".format(
    DATASET, OUTPUT_MAP_SIZE, SEED, MASKING_MODE, MASKING_SHAPE
)


def precompute_dataset_for_map(kwargs):
    cfg = kwargs["cfg"]
    split = kwargs["split"]
    name = kwargs["name"]
    n_samples_per_map = kwargs["n_samples_per_map"]
    save_root = kwargs["save_root"]

    dataset = SemanticMapDataset(
        cfg.DATASET, split=split, scf_name=name, seed=SEED
    )
    print(f'====> Pre-computing for map {name}')
    os.makedirs(f'{save_root}/{name}', exist_ok=True)
    for i in range(n_samples_per_map):
        input, label = dataset.get_item_by_name(name)
        save_path = f'{save_root}/{name}/sample_{i:05d}.pbz2'
        in_semmap = asnumpy(input) > 0.5 # (N, H, W)
        semmap = asnumpy(label['semmap']) > 0.5 # (N, H, W)
        fmm_dists = asnumpy(label['fmm_dists']).astype(np.float32) # (N, H, W)
        world_xyz = np.array(label['world_xyz']) # (3, )
        world_heading = np.array([label['world_heading']]) # (1, )
        scene_name = np.array(label['scene_name'])
        # Convert to int maps to save space
        fmm_dists = (fmm_dists * 100.0).astype(np.int32)
        with bz2.BZ2File(save_path, 'w') as fp:
            cPickle.dump(
                {
                    'in_semmap': in_semmap,
                    'semmap': semmap,
                    'fmm_dists': fmm_dists,
                    'scene_name': scene_name,
                    'world_xyz': world_xyz,
                    'world_heading': world_heading,
                },
                fp
            )


def precompute_dataset(args):
    cfg = get_cfg()
    cfg.defrost()
    cfg.SEED = SEED
    cfg.DATASET.dset_name = DATASET
    cfg.DATASET.root = DATA_ROOT
    cfg.DATASET.output_map_size = OUTPUT_MAP_SIZE
    cfg.DATASET.fmm_dists_saved_root = FMM_DISTS_SAVED_ROOT
    cfg.DATASET.masking_mode = MASKING_MODE
    cfg.DATASET.masking_shape = MASKING_SHAPE
    cfg.DATASET.visibility_size = 3.0 # m
    cfg.freeze()

    os.makedirs(SAVE_ROOT, exist_ok=True)

    os.makedirs(os.path.join(SAVE_ROOT, args.split), exist_ok=True)
    dataset = SemanticMapDataset(cfg.DATASET, split=args.split)
    n_maps = len(dataset)
    print('Maps', n_maps)
    n_samples_per_map = (NUM_SAMPLES[args.split] // n_maps) + 1

    if args.map_id != -1:
        map_names = [dataset.names[args.map_id]]
    elif args.map_id_range is not None:
        assert len(args.map_id_range) == 2
        map_names = [
            dataset.names[i]
            for i in range(args.map_id_range[0], args.map_id_range[1] + 1)
        ]
    else:
        map_names = dataset.names

    pool = mp.Pool(processes=args.num_workers)
    inputs = []
    for name in map_names:
        kwargs = {
            "cfg": cfg,
            "split": args.split,
            "name": name,
            "n_samples_per_map": n_samples_per_map,
            "save_root": f'{SAVE_ROOT}/{args.split}',
        }
        inputs.append(kwargs)
    
    with tqdm.tqdm(total=len(inputs)) as pbar:
        for _ in pool.imap_unordered(precompute_dataset_for_map, inputs):
            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-id', type=int, default=-1)
    parser.add_argument('--map-id-range', type=int, nargs="+", default=None)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=25)
    args = parser.parse_args()

    # Both map-id and map-id-range should not be enabled simultaneously
    assert (args.map_id == -1) or (args.map_id_range is None)

    precompute_dataset(args)
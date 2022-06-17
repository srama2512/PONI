import _pickle as cPickle
import bz2
import multiprocessing as mp
import os

import tqdm
from poni.dataset import SemanticMapDataset
from poni.default import get_cfg


assert "ACTIVE_DATASET" in os.environ
ACTIVE_DATASET = os.environ["ACTIVE_DATASET"]
SEED = 123
DATA_ROOT = "data/semantic_maps/{}/semantic_maps".format(ACTIVE_DATASET)
SAVE_ROOT = "data/semantic_maps/{}/fmm_dists_{}".format(ACTIVE_DATASET, SEED)
NUM_WORKERS = 24


def save_data(inputs):
    data, path = inputs
    with bz2.BZ2File(path, "w") as fp:
        cPickle.dump(data, fp)


def precompute_fmm_dists():
    cfg = get_cfg()
    cfg.defrost()
    cfg.SEED = SEED
    cfg.DATASET.dset_name = ACTIVE_DATASET
    cfg.DATASET.root = DATA_ROOT
    cfg.DATASET.fmm_dists_saved_root = ""
    cfg.freeze()
    os.makedirs(SAVE_ROOT, exist_ok=True)
    pool = mp.Pool(NUM_WORKERS)

    for split in ["val", "train"]:
        print(f"=====> Computing FMM dists for {split} split")
        dataset = SemanticMapDataset(cfg.DATASET, split=split)
        print("--> Saving FMM dists")
        inputs = []
        for name in dataset.names:
            save_path = os.path.join(SAVE_ROOT, f"{name}.pbz2")
            data = dataset.fmm_dists[name]
            inputs.append((data, save_path))
        _ = list(tqdm.tqdm(pool.imap(save_data, inputs), total=len(inputs)))


if __name__ == "__main__":
    precompute_fmm_dists()

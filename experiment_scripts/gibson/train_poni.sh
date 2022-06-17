#!/bin/bash

EXPT_ROOT=$PWD

conda activate poni

cd $PONI_ROOT

python -W ignore -u -m torch.distributed.launch \
  --use_env \
  --nproc_per_node=2 \
  train.py \
    LOGGING.ckpt_dir "$EXPT_ROOT/checkpoints" \
    LOGGING.tb_dir "$EXPT_ROOT/tb" \
    LOGGING.log_dir "$EXPT_ROOT" \
    DATASET.root $PONI_ROOT/data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square \
    DATASET.enable_unexp_area True

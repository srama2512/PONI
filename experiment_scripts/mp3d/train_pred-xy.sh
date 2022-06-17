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
    DATASET.root $PONI_ROOT/data/semantic_maps/mp3d/precomputed_dataset_24.0_123_spath_square \
    DATASET.dset_name 'mp3d' \
    DATASET.enable_locations True \
    OPTIM.batch_size 20 \
    MODEL.num_categories 23 \
    MODEL.object_loss_type 'l2' \
    MODEL.object_activation 'sigmoid'

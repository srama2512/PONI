#!/bin/bash

export GLOG_minloglevel=2 MAGNUM_LOG=quiet
export MODEL_PATH=$PONI_ROOT/pretrained_models/gibson_models/pred_act_seed_123.ckpt
export PYTHONPATH=$PYTHONPATH:$PONI_ROOT

export EXPT_ROOT=$PWD
cd $PONI_ROOT/semexp

conda activate poni

python eval_poni.py \
  --split val \
  --seed 100 \
  --eval 1 \
  --pf_model_path $MODEL_PATH \
  -d $EXPT_ROOT/gibson_objectnav \
  --num_local_steps 1 \
  --exp_name "seed_100" \
  --global_downscaling 1 \
  --mask_nearest_locations

#!/bin/bash

EXPT_ROOT=$PWD

export PYTHONPATH=$PYTHONPATH:$PONI_ROOT
export PYTHONPATH=$PYTHONPATH:$PONI_ROOT/dependencies/astar_pycpp
export MAGNUM_LOG=quiet GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet

MODEL_PATH="../pretrained_models/mp3d_models/pred_xy_seed_123.ckpt"

SAVE_ROOT=$EXPT_ROOT/mp3d_objectnav
mkdir -p $SAVE_ROOT

cd $PONI_ROOT/hlab

conda activate poni

DEVICE_1=$1
DEVICE_2=$2

for part_id in 0 1 2 3 4 5; do
    val_part="val_part_${part_id}"
    CUDA_VISIBLE_DEVICES="$DEVICE_1" python eval_poni.py \
    --exp-config transfer_configs/transfer_objectnav_mp3d.yaml \
    TASK_CONFIG.DATASET.DATA_PATH "../data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz" \
    EVAL.SPLIT $val_part \
    TASK_CONFIG.SEED 100 \
    TENSORBOARD_DIR $SAVE_ROOT/tb_seed_100_${val_part} \
    LOG_FILE $SAVE_ROOT/logs_seed_100_${val_part}.txt \
    GLOBAL_AGENT.name "PFExp" \
    GLOBAL_AGENT.smart_local_boundaries True \
    PF_EXP_POLICY.pf_model_path $MODEL_PATH &
done

for part_id in 6 7 8 9 10; do
    val_part="val_part_${part_id}"
    CUDA_VISIBLE_DEVICES="$DEVICE_2" python eval_poni.py \
    --exp-config transfer_configs/transfer_objectnav_mp3d.yaml \
    TASK_CONFIG.DATASET.DATA_PATH "../data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz" \
    EVAL.SPLIT $val_part \
    TASK_CONFIG.SEED 100 \
    TENSORBOARD_DIR $SAVE_ROOT/tb_seed_100_${val_part} \
    LOG_FILE $SAVE_ROOT/logs_seed_100_${val_part}.txt \
    GLOBAL_AGENT.name "PFExp" \
    GLOBAL_AGENT.smart_local_boundaries True \
    PF_EXP_POLICY.pf_model_path $MODEL_PATH &
done

wait
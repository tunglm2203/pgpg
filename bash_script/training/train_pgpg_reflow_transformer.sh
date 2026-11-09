#!/bin/bash
# PGPG fine-tuning of a pretrained policy.
# Example:
#   GPU=0 TASK=square bash bash_script/training/train_pgpg_reflow_transformer.sh
#   GPU=0 bash bash_script/training/train_pgpg_reflow_transformer.sh   # all five tasks in turn

PROJECT=${WANDB_PROJECT:-GBC-PBRL-2026}
ENTITY=${WANDB_ENTITY:-}
GPU=${GPU:-0}
SEED=${SEED:-42}
NUM_EPOCHS=${NUM_EPOCHS:-1000}
TASKS=${TASK:-"lift can square transport tool_hang"}

CONFIG_DIR="diffusion_policy/config/pbrl_reflow_transformer"

# shared parameter
CPL_TYPE="cplkl"
BIAS_REG=0.25
USE_EXP_DATA_1=1
USE_EXP_DATA_2=0
DENSE_REWARD=1
IGNORE_EQUAL_PREF=0
N_EPOCH_SFT=0
SFT_TYPE="pos"
STRIDE=1
MIN_PROGRESS=0
CONFIDENCE_WEIGHT=0
VAL_RATIO=0.6
VAL_RATIO_2=0.0
BATCH_SIZE=128

for TASK in $TASKS; do
  # SEG_SIZE  N_QUERIES  SEG_MARGIN  N_DEMOS  EQUAL_THR  CPL_BETA  CLIP_MARGIN  SMOOTHNESS  UNCLIP_WIN
  case "$TASK" in
    lift)      SEG_SIZE=100; N_QUERIES=10000; SEG_MARGIN=0.6; N_DEMOS_FOR_PREF=10
               EQUAL_THRESHOLD=0.05
               CPL_BETA=0.01
               CLIP_MARGIN=0.3
               SMOOTHNESS=0
               UNCLIP_WIN=0 ;;
    can)       SEG_SIZE=100; N_QUERIES=20000;  SEG_MARGIN=0.3; N_DEMOS_FOR_PREF=20
               EQUAL_THRESHOLD=0
               CPL_BETA=0.01
               CLIP_MARGIN=0.3
               SMOOTHNESS=0
               UNCLIP_WIN=1 ;;
    square)    SEG_SIZE=200; N_QUERIES=20000; SEG_MARGIN=0.3; N_DEMOS_FOR_PREF=40
               EQUAL_THRESHOLD=0
               CPL_BETA=0.05
               CLIP_MARGIN=0.5
               SMOOTHNESS=0.1
               UNCLIP_WIN=1 ;;
    transport) SEG_SIZE=400; N_QUERIES=20000; SEG_MARGIN=0.4; N_DEMOS_FOR_PREF=40
               EQUAL_THRESHOLD=0
               CPL_BETA=1.0
               CLIP_MARGIN=0.3
               SMOOTHNESS=0.1
               UNCLIP_WIN=1 ;;
    tool_hang) SEG_SIZE=300; N_QUERIES=20000; SEG_MARGIN=0.3; N_DEMOS_FOR_PREF=40
               EQUAL_THRESHOLD=0
               CPL_BETA=0.5
               CLIP_MARGIN=0.2
               SMOOTHNESS=0.15
               UNCLIP_WIN=1 ;;
    *) echo "Task '$TASK' is not supported."; exit 1 ;;
  esac

  # tool_hang ships proficient-human demos; the rest are multi-human
  DEMO_DIR=$([ "$TASK" = tool_hang ] && echo demo_ph || echo demo_mh)
  SUFFIX=$([ "$TASK" = tool_hang ] && echo ph || echo mh)
  CONFIG_NAME="train_pbrl_reflow_policy_transformer_lowdim_${TASK}_${SUFFIX}_abs.yaml"
  DATASET_PATH="logs/collect_data/\${task_name}/${DEMO_DIR}/low_dim_abs.hdf5"
  ROLLOUTS="logs/collect_data/${TASK}_lowdim/datacollect_diffusion_transformer_lowdim_${TASK}_lowdim/collect_${TASK}_lowdim.hdf5"

  EXP_NAME="pgpg_flow_policy"

  WANDB_ARGS=(logging.project="${PROJECT}" logging.mode="${WANDB_MODE:-disabled}")
  [ -n "$ENTITY" ] && WANDB_ARGS+=(+logging.entity="${ENTITY}")

  echo "=============================================================================="
  echo " PGPG (rectified flow)  task=${TASK}  seed=${SEED}  epochs=${NUM_EPOCHS}  GPU=${GPU}"
  echo " N=${N_QUERIES} L=${SEG_SIZE} segM=${SEG_MARGIN} nD=${N_DEMOS_FOR_PREF} thr=${EQUAL_THRESHOLD}"
  echo " beta=${CPL_BETA} clip=${CLIP_MARGIN} smt=${SMOOTHNESS} unclipW=${UNCLIP_WIN}"
  echo "=============================================================================="

  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --config-dir="${CONFIG_DIR}" \
    --config-name="${CONFIG_NAME}" \
    training.resume=True \
    name="${EXP_NAME}" \
    "${WANDB_ARGS[@]}" \
    training.rollout_every=50 training.checkpoint_every=50 \
    training.cpl_loss_type="${CPL_TYPE}" \
    training.use_expert_data_1=${USE_EXP_DATA_1} training.use_expert_data_2=${USE_EXP_DATA_2} \
    training.dataset_1_dir=${ROLLOUTS} training.dataset_2_dir=${ROLLOUTS} \
    task.dataset_path="${DATASET_PATH}" task.dense_reward=${DENSE_REWARD} \
    task.pref_dataset.num_queries=${N_QUERIES} task.pref_dataset.sequence_length=${SEG_SIZE} \
    task.pref_dataset.val_ratio_data1=${VAL_RATIO} task.pref_dataset.val_ratio_data2=${VAL_RATIO_2} \
    task.pref_dataset.n_demos_for_preference=${N_DEMOS_FOR_PREF} \
    task.pref_dataset.seg_margin=${SEG_MARGIN} task.pref_dataset.min_progress=${MIN_PROGRESS} \
    policy.bias_reg=${BIAS_REG} \
    policy.ignore_equal_pref=${IGNORE_EQUAL_PREF} \
    policy.beta=${CPL_BETA} \
    policy.clip_margin=${CLIP_MARGIN} \
    policy.smooth_label=${SMOOTHNESS} \
    policy.confidence_weight=${CONFIDENCE_WEIGHT} \
    policy.unclip_win=${UNCLIP_WIN} \
    training.preference_learning.equal_threshold=${EQUAL_THRESHOLD} \
    training.n_epoch_sft=${N_EPOCH_SFT} training.sft_type=${SFT_TYPE} \
    training.stride_ratio=${STRIDE} \
    training.pseudo_preference=True \
    dataloader.batch_size=${BATCH_SIZE} \
    training.seed=${SEED} \
    training.num_epochs=${NUM_EPOCHS} \
    training.debug=False \
    hydra.run.dir='logs/pbrl/${task_name}/${logging.group}/${logging.name}'
done

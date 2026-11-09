#!/bin/bash
# Evaluate a PGPG-finetuned rectified-flow policy.
# Example:
#   GPU=0 TASK=square bash bash_script/evaluation/eval_pgpg_reflow_transformer.sh
#   GPU=0 bash bash_script/evaluation/eval_pgpg_reflow_transformer.sh   # all five tasks in turn

GPU=${GPU:-0}
SEED=${SEED:-42}
TASKS=${TASK:-"lift can square transport tool_hang"}
OUTDIR=${OUTDIR:-eval_logs}

# shared parameter
EXP_NAME="pgpg_flow_policy"
CKPT_SELECTION=${CKPT_SELECTION:-last}
NTEST=${NTEST:-50}
EVAL_SEEDS=${EVAL_SEEDS:-"1 2 3"}
USE_EMA=False
N_INFER_STEPS=8

for TASK in $TASKS; do
  case "$TASK" in
    lift|can|square|transport|tool_hang) ;;
    *) echo "Task '$TASK' is not supported."; exit 1 ;;
  esac

  CKPT="logs/pbrl/${TASK}_lowdim/${EXP_NAME}"
  if [ ! -d "$CKPT" ]; then
    echo "No run at ${CKPT} -- train it first with bash_script/training/train_pgpg_reflow_transformer.sh"
    continue
  fi

  echo "=============================================================================="
  echo " Eval PGPG (rectified flow)  task=${TASK}  ckpt=${CKPT_SELECTION}  ntest=${NTEST}  GPU=${GPU}"
  echo "=============================================================================="

  for s in ${EVAL_SEEDS}; do
    CUDA_VISIBLE_DEVICES=${GPU} python eval_reflow_policy_pbrl.py \
      --checkpoint "${CKPT}" \
      --output_dir "${OUTDIR}/noise_0.0" \
      --ckpt_selection ${CKPT_SELECTION} \
      --ntest ${NTEST} \
      --use_ema ${USE_EMA} --num_inference_steps ${N_INFER_STEPS} \
      --seed ${s}
  done
done

#!/bin/bash
# Evaluate a PGPG-finetuned diffusion policy.
# Example:
#   GPU=0 TASK=square bash bash_script/evaluation/eval_pgpg_dp_transformer.sh
#   GPU=0 bash bash_script/evaluation/eval_pgpg_dp_transformer.sh   # all five tasks in turn

GPU=${GPU:-0}
SEED=${SEED:-42}
TASKS=${TASK:-"lift can square transport tool_hang"}
OUTDIR=${OUTDIR:-eval_logs}

# shared parameter
EXP_NAME="pgpg_diffusion_policy"
CKPT_SELECTION=${CKPT_SELECTION:-last}
NTEST=${NTEST:-50}
EVAL_SEEDS=${EVAL_SEEDS:-"1 2 3"}
NOISE_SCHEDULER="ddpm"
N_INFER_STEPS=100

for TASK in $TASKS; do
  case "$TASK" in
    lift|can|square|transport|tool_hang) ;;
    *) echo "Task '$TASK' is not supported."; exit 1 ;;
  esac

  CKPT="logs/pbrl/${TASK}_lowdim/${EXP_NAME}"
  if [ ! -d "$CKPT" ]; then
    echo "No run at ${CKPT} -- train it first with bash_script/training/train_pgpg_dp_transformer.sh"
    continue
  fi

  echo "=============================================================================="
  echo " Eval PGPG (diffusion)  task=${TASK}  ckpt=${CKPT_SELECTION}  ntest=${NTEST}  GPU=${GPU}"
  echo "=============================================================================="

  for s in ${EVAL_SEEDS}; do
    CUDA_VISIBLE_DEVICES=${GPU} python eval_diffusion_policy_pbrl.py \
      --checkpoint "${CKPT}" \
      --output_dir "${OUTDIR}/noise_0.0" \
      --ckpt_selection ${CKPT_SELECTION} \
      --ntest ${NTEST} \
      --noise_scheduler ${NOISE_SCHEDULER} --num_inference_steps ${N_INFER_STEPS} \
      --seed ${s}
  done
done

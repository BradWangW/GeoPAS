#!/usr/bin/env bash
set -euo pipefail

# Minimal runner for train.py (multi-GPU parallel CV orchestrator).
# Usage examples:
#   bash train.sh
#   PROTOCOL=all GPUS=0,1,2,3 MAX_PARALLEL=4 bash train.sh
#   PROTOCOL=lpo CSV=data/log_relert_bbob.csv OUT_DIR=preds_parallel/lpo_$(date +%Y%m%d_%H%M%S) bash train.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data1/home/jw1017/miniforge3/envs/as_bbo/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "ERROR: PYTHON_BIN not found/executable: $PYTHON_BIN" >&2
	echo "Set PYTHON_BIN to your as_bbo python, e.g.:" >&2
	echo "  PYTHON_BIN=/path/to/miniforge3/envs/as_bbo/bin/python bash train.sh" >&2
	exit 1
fi

PROTOCOL="lio"                  # "lpo", "lio", or "random"
CSV="data/relert_bbob.csv"      # protocol CSV file
DATA_ROOT="data/bbob"
RESOLUTION=16
K_VIEWS=16
NUM_REPETITIONS="10"

BATCH_SIZE="16"
NUM_EPOCHS="50"
LR="1e-3"
WEIGHT_DECAY="1e-5"
NUM_WORKERS="4"

SEED="42"
TEST_RATIO="0.2"
N_SPLITS="5"

GPUS="auto"                 # "auto" or comma-separated GPU indices (e.g. 0,1,2,3)
MAX_PARALLEL=""             # empty = default to number of GPUs
OUT_DIR_BASE="results/bbob"

CACHE_TRAIN="0"              # 1 to cache train samples in RAM per worker
CACHE_TEST="0"               # 1 to cache test samples in RAM per worker

# Model/score options
# Model/score sweep options
TAIL_PENALTY_LIST=(1)              # 1: enable tail penalty, 0: disable
TAIL_LAM_CAP_LIST=(5.0 10.0 15.0 20.0)
TAIL_LAM_THR_LIST=(3.0 5.0 10.0)
TAIL_SCALE="1.0"

DUAL_HEAD_LIST=(1)                # 1: regression+cat head, 0: regression only
CAT_LOSS_WEIGHT_LIST=(15.0 20.0)
CAT_TAU_LIST=(0.3)
CAT_PENALTY_LIST=(15.0 20.0)

echo "Running train.py"
echo "  python:     $PYTHON_BIN"
echo "  protocol:   $PROTOCOL"
echo "  csv:        $CSV"
echo "  data_root:  $DATA_ROOT" 
echo "  gpus:       $GPUS"
echo "  dual_head:  $DUAL_HEAD_LIST"
echo "  tail_pen:   $TAIL_PENALTY_LIST"

for tail_penalty in "${TAIL_PENALTY_LIST[@]}"; do
	for tail_lam_cap in "${TAIL_LAM_CAP_LIST[@]}"; do
		for tail_lam_thr in "${TAIL_LAM_THR_LIST[@]}"; do
			for dual_head in "${DUAL_HEAD_LIST[@]}"; do
				for cat_loss_weight in "${CAT_LOSS_WEIGHT_LIST[@]}"; do
					for cat_tau in "${CAT_TAU_LIST[@]}"; do
						for cat_penalty in "${CAT_PENALTY_LIST[@]}"; do
							extra_flags=()
							if [[ "$CACHE_TRAIN" == "1" ]]; then extra_flags+=(--cache-train); fi
							if [[ "$CACHE_TEST" == "1" ]]; then extra_flags+=(--cache-test); fi
							if [[ -n "$MAX_PARALLEL" ]]; then extra_flags+=(--max-parallel "$MAX_PARALLEL"); fi

							if [[ "$dual_head" == "1" ]]; then
								extra_flags+=(--dual-head)
								extra_flags+=(--cat-loss-weight "$cat_loss_weight" --cat-tau "$cat_tau" --cat-penalty "$cat_penalty")
							else
								extra_flags+=(--single-head)
							fi

							if [[ "$tail_penalty" == "1" ]]; then
								extra_flags+=(--tail-penalty --tail-lam-cap "$tail_lam_cap" --tail-lam-thr "$tail_lam_thr" --tail-scale "$TAIL_SCALE")
							else
								extra_flags+=(--no-tail-penalty)
							fi

							run_out_dir="${OUT_DIR_BASE}/tail${tail_penalty}_cap${tail_lam_cap}_thr${tail_lam_thr}_dual${dual_head}_w${cat_loss_weight}_tau${cat_tau}_pen${cat_penalty}"
							mkdir -p "$run_out_dir"

							echo "Starting: tail_penalty=$tail_penalty tail_lam_cap=$tail_lam_cap tail_lam_thr=$tail_lam_thr dual_head=$dual_head cat_loss_weight=$cat_loss_weight cat_tau=$cat_tau cat_penalty=$cat_penalty"

							"$PYTHON_BIN" -u "$ROOT_DIR/train_parallel.py" \
								--protocol "$PROTOCOL" \
								--csv "$CSV" \
								--data-root "$DATA_ROOT" \
								--resolution "$RESOLUTION" \
								--k-views "$K_VIEWS" \
								--num-repetitions "$NUM_REPETITIONS" \
								--batch-size "$BATCH_SIZE" \
								--num-epochs "$NUM_EPOCHS" \
								--lr "$LR" \
								--weight-decay "$WEIGHT_DECAY" \
								--num-workers "$NUM_WORKERS" \
								--seed "$SEED" \
								--test-ratio "$TEST_RATIO" \
								--n-splits "$N_SPLITS" \
								--gpus "$GPUS" \
								--out-dir "$run_out_dir" \
								"${extra_flags[@]}"
						done
					done
				done
			done
		done
	done
done

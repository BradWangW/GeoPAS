#!/usr/bin/env bash
set -euo pipefail

# Minimal runner for train.py (multi-GPU parallel CV orchestrator).
# Usage examples:
#   bash train.sh
#   PROTOCOL=all GPUS=0,1,2,3 MAX_PARALLEL=4 bash train.sh
#   SKIP_EXISTING=1 bash train.sh
#   # NOTE: `output_results` expects relERT (>= 1). If you pass log-domain CSVs here,
#   # your reported AS can be < 1 and you may be mixing domains.
#   PROTOCOL=lpo CSV=data/relert_bbob.csv OUT_DIR=preds_parallel/lpo_$(date +%Y%m%d_%H%M%S) bash train.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data1/home/jw1017/miniforge3/envs/as_bbo/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "ERROR: PYTHON_BIN not found/executable: $PYTHON_BIN" >&2
	echo "Set PYTHON_BIN to your as_bbo python, e.g.:" >&2
	echo "  PYTHON_BIN=/path/to/miniforge3/envs/as_bbo/bin/python bash train.sh" >&2
	exit 1
fi

ELL_MAX="0.7"
LOG_UNIFORM_SCALE="true"

PROTOCOLS=("lpo" "lio" "random")                # "lpo", "lio", or "random"
CSV="/data1/home/jw1017/AS_BBO_REBUILT/data/bbob_by_deepela/relert.csv"      # protocol CSV file
DATA_ROOT="/data1/home/jw1017/AS_BBO_REBUILT/data/bbob_by_deepela//maxscale_${ELL_MAX}_logscale_${LOG_UNIFORM_SCALE}/"
RESOLUTIONS=(8)
KS_VIEWS=(32)
NUM_REPETITIONS="10"

BATCH_SIZE="16"
NUM_EPOCHS="50"
LR="1e-3"
WEIGHT_DECAY="0"
NUM_WORKERS="3"
EARLY_STOPPING_PATIENCE="100"

SEED=17
TEST_RATIO="0.2"
N_SPLITS="5"

GPUS="auto"                 # "auto" or comma-separated GPU indices (e.g. 0,1,2,3)
MAX_PARALLEL=""             # empty = default to number of GPUs
OUT_DIR_BASE="/data1/home/jw1017/AS_BBO_REBUILT/results/bbob_by_deepela/results/bbob"
SKIP_EXISTING="${SKIP_EXISTING:-0}"   # 1: skip configs whose final results CSV already exists

# TensorBoard logging
TB_LOG_DIR="/data1/home/jw1017/AS_BBO_REBUILT/results/bbob_by_deepela/results/tensorboard"     # empty string disables TensorBoard logging
TB_LOG_VAL="1"                       # 1: log val/as, 0: disable val/as logging

CACHE_TRAIN="0"              # 1 to cache train samples in RAM per worker
CACHE_TEST="0"               # 1 to cache test samples in RAM per worker

# Model/score options
# Model/score sweep options
TARGET_SCALE_LIST=(log)        # 'log' trains on log(relERT); 'raw' trains on relERT
TAIL_PENALTY_LIST=(1)              # 1: enable tail penalty, 0: disable
TAIL_LAM_CAP_LIST=(3.0 4.0 5.0)
TAIL_LAM_THR_LIST=(5.0 6.0 7.0)
TAIL_SCALE_LIST=(1.0)

DUAL_HEAD_LIST=(1)                # 1: regression+cat head, 0: regression only
# CAT_LOSS_WEIGHT_LIST=(5.0 10.0 15.0)
# CAT_TAU_LIST=(0.2 0.3)
# CAT_PENALTY_LIST=(10.0 15.0 20.0)
CAT_LOSS_WEIGHT_LIST=(5.0 10.0)
CAT_TAU_LIST=(0.3 0.5)
CAT_PENALTY_LIST=(5.0 10.0 15.0)

# Model/score options
# Model/score sweep options
# TARGET_SCALE_LIST=(log)        # 'log' trains on log(relERT); 'raw' trains on relERT
# TAIL_PENALTY_LIST=(1)              # 1: enable tail penalty, 0: disable
# TAIL_LAM_CAP_LIST=(1.0 3.0 5.0)
# TAIL_LAM_THR_LIST=(1.0 3.0 5.0)
# TAIL_SCALE="1.0"
# DUAL_HEAD_LIST=(1)                # 1: regression+cat head, 0: regression only
# CAT_LOSS_WEIGHT_LIST=(5.0 10.0 15.0)
# CAT_TAU_LIST=(0.2 0.3 0.4)
# CAT_PENALTY_LIST=(5.0 10.0 15.0)

echo "Running train.py"
echo "  python:     $PYTHON_BIN"
echo "  protocols:   $PROTOCOLS"
echo "  csv:        $CSV"
echo "  data_root:  $DATA_ROOT" 
echo "  gpus:       $GPUS"
echo "  dual_head:  $DUAL_HEAD_LIST"
echo "  tail_pen:   $TAIL_PENALTY_LIST"
echo "  skip_existing: $SKIP_EXISTING"
echo "Total number of runs: $(( ${#PROTOCOLS[@]} * ${#RESOLUTIONS[@]} * ${#KS_VIEWS[@]} * ${#TARGET_SCALE_LIST[@]} * ${#TAIL_PENALTY_LIST[@]} * ${#TAIL_LAM_CAP_LIST[@]} * ${#TAIL_LAM_THR_LIST[@]} * ${#DUAL_HEAD_LIST[@]} * ${#CAT_LOSS_WEIGHT_LIST[@]} * ${#CAT_TAU_LIST[@]} * ${#CAT_PENALTY_LIST[@]} ))"

for PROTOCOL in "${PROTOCOLS[@]}"; do
	case "$PROTOCOL" in
		lpo) protocol_dir_name="LPO" ;;
		lio) protocol_dir_name="LIO" ;;
		random) protocol_dir_name="RANDOM" ;;
		*)
			echo "ERROR: Unsupported protocol: $PROTOCOL" >&2
			exit 1
			;;
	esac

	for RESOLUTION in "${RESOLUTIONS[@]}"; do
		for K_VIEWS in "${KS_VIEWS[@]}"; do
			for target_scale in "${TARGET_SCALE_LIST[@]}"; do
				for tail_penalty in "${TAIL_PENALTY_LIST[@]}"; do
					for tail_lam_cap in "${TAIL_LAM_CAP_LIST[@]}"; do
						for tail_lam_thr in "${TAIL_LAM_THR_LIST[@]}"; do
							for tail_scale in "${TAIL_SCALE_LIST[@]}"; do
								for dual_head in "${DUAL_HEAD_LIST[@]}"; do
									for cat_loss_weight in "${CAT_LOSS_WEIGHT_LIST[@]}"; do
										for cat_tau in "${CAT_TAU_LIST[@]}"; do
											for cat_penalty in "${CAT_PENALTY_LIST[@]}"; do
												run_tag="scale${target_scale}_tail${tail_penalty}_cap${tail_lam_cap}_thr${tail_lam_thr}_dual${dual_head}_w${cat_loss_weight}_tau${cat_tau}_pen${cat_penalty}"
												run_tag_with_tail_scale="scale${target_scale}_tail${tail_penalty}_cap${tail_lam_cap}_thr${tail_lam_thr}_scale${tail_scale}_dual${dual_head}_w${cat_loss_weight}_tau${cat_tau}_pen${cat_penalty}"
												extra_flags=()
												if [[ "$CACHE_TRAIN" == "1" ]]; then extra_flags+=(--cache-train); fi
												if [[ "$CACHE_TEST" == "1" ]]; then extra_flags+=(--cache-test); fi
												if [[ -n "$MAX_PARALLEL" ]]; then extra_flags+=(--max-parallel "$MAX_PARALLEL"); fi
												if [[ -n "$TB_LOG_DIR" ]]; then
													run_tb_dir="${TB_LOG_DIR}/${run_tag}/${PROTOCOL}"
													extra_flags+=(--tb-log-dir "$run_tb_dir")
													if [[ "$TB_LOG_VAL" == "1" ]]; then
														extra_flags+=(--tb-log-val)
													else
														extra_flags+=(--no-tb-log-val)
													fi
												fi

												if [[ "$dual_head" == "1" ]]; then
													extra_flags+=(--dual-head)
													extra_flags+=(--cat-loss-weight "$cat_loss_weight" --cat-tau "$cat_tau" --cat-penalty "$cat_penalty")
												else
													extra_flags+=(--single-head)
												fi

												if [[ "$tail_penalty" == "1" ]]; then
													extra_flags+=(--tail-penalty --tail-lam-cap "$tail_lam_cap" --tail-lam-thr "$tail_lam_thr" --tail-scale "$tail_scale")
												else
													extra_flags+=(--no-tail-penalty)
												fi


												run_out_dir="${OUT_DIR_BASE}/${run_tag_with_tail_scale}/${PROTOCOL}/seed${SEED}"
												final_csv="${run_out_dir}/${protocol_dir_name}/res_tailpenalty_${tail_penalty}_taillamcap_${tail_lam_cap}_taillamthr_${tail_lam_thr}_catlossweight_${cat_loss_weight}_cattau_${cat_tau}_catpenalty_${cat_penalty}_res_${RESOLUTION}_k_views_${K_VIEWS}.csv"

														if [[ "$SKIP_EXISTING" == "1" && -f "$final_csv" ]]; then
															echo "Skipping existing config: $final_csv"
															continue
														fi

												mkdir -p "$run_out_dir"

													echo "Starting: target_scale=$target_scale tail_penalty=$tail_penalty tail_lam_cap=$tail_lam_cap tail_lam_thr=$tail_lam_thr tail_scale=$tail_scale dual_head=$dual_head cat_loss_weight=$cat_loss_weight cat_tau=$cat_tau cat_penalty=$cat_penalty"

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
													--early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
													--seed "$SEED" \
													--test-ratio "$TEST_RATIO" \
													--n-splits "$N_SPLITS" \
													--gpus "$GPUS" \
													--target-scale "$target_scale" \
													--out-dir "$run_out_dir" \
													"${extra_flags[@]}"
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

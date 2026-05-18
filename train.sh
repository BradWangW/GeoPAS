#!/usr/bin/env bash
set -euo pipefail

# Two-phase runner for train_parallel.py.
# 1) Train one network per base configuration and persist reusable base scores.
# 2) Materialize one or more scoring variants from those saved base scores.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROJECT_ROOT="${PROJECT_ROOT:-${GEOPAS_PROJECT_ROOT:-$(cd "$ROOT_DIR/.." && pwd)}}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results/bbob_by_deepela/results}"
RELERT_CSV_DEFAULT="$ROOT_DIR/data_generation/performances/relert.csv"

DEFAULT_PYTHON_BIN="/data1/home/jw1017/miniforge3/envs/as_bbo/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
if [[ ! -x "$PYTHON_BIN" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
	PYTHON_BIN="$CONDA_PREFIX/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "ERROR: PYTHON_BIN not found/executable: $PYTHON_BIN" >&2
	echo "Set PYTHON_BIN to your as_bbo python, e.g.:" >&2
	echo "  PYTHON_BIN=/path/to/miniforge3/envs/as_bbo/bin/python bash train.sh" >&2
	exit 1
fi

ELL_MAX="0.7"
LOG_UNIFORM_SCALE="false"

PROTOCOLS=("lpo" "lio" "random")                # "lpo", "lio", or "random"
CSV="${CSV:-$RELERT_CSV_DEFAULT}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/bbob_by_deepela/maxscale_${ELL_MAX}_logscale_${LOG_UNIFORM_SCALE}}"
RESOLUTIONS=(8)
KS_VIEWS=(32)
NUM_REPETITIONS="10"

BATCH_SIZE="16"
NUM_EPOCHS="50"
LR="1e-3"
WEIGHT_DECAY="3e-4"
NUM_WORKERS="3"
EARLY_STOPPING_ENABLED="0"
EARLY_STOPPING_PATIENCE="10"
VAL_RATIO_LPO="0.0"
VAL_RATIO_LIO="0.1"
VAL_RATIO_RANDOM="0.1"

SEEDS="${SEEDS:-16,17,18}"
TEST_RATIO="0.2"
N_SPLITS="5"

GPUS="auto"
JOBS_PER_GPU="1"
MAX_PARALLEL=""
OUT_DIR_BASE="${OUT_DIR_BASE:-$RESULTS_ROOT/bbob}"
BASE_SCORES_ROOT_DEFAULT="$RESULTS_ROOT/base_scores/bbob"
BASE_SCORES_ROOT="${BASE_SCORES_ROOT:-$BASE_SCORES_ROOT_DEFAULT}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

TB_LOG_DIR_DEFAULT="$RESULTS_ROOT/tensorboard"
TB_LOG_DIR="${TB_LOG_DIR-$TB_LOG_DIR_DEFAULT}"
TB_LOG_VAL="1"

CACHE_TRAIN="0"
CACHE_TEST="0"

TARGET_SCALE_LIST=(log_power)
HEAD_2_TARGET_SCALE_LIST=(norm)
SIGMOID_LOG_S_LIST=(1.0)
PRIOR_SCALE_LIST=(log_power)
TAIL_SCALE_LIST=(1.0)
LAM_PRIOR_LIST=(0.5)

DUAL_HEAD_LIST=(0)
HEAD_2_LOSS_WEIGHT_LIST=(0.0)
HEAD_2_SCORE_WEIGHT_LIST=(0.0)

effective_val_ratio_lpo="$VAL_RATIO_LPO"
effective_val_ratio_lio="$VAL_RATIO_LIO"
effective_val_ratio_random="$VAL_RATIO_RANDOM"
if [[ "$EARLY_STOPPING_ENABLED" != "1" ]]; then
	effective_val_ratio_lpo="0.0"
	effective_val_ratio_lio="0.0"
	effective_val_ratio_random="0.0"
fi

SEED_LIST=()
for seed in ${SEEDS//,/ }; do
	if [[ -n "$seed" ]]; then
		SEED_LIST+=("$seed")
	fi
done
SEED_COUNT="${#SEED_LIST[@]}"

join_by_comma() {
	local IFS="," 
	printf '%s' "$*"
}

run_train_parallel() {
	local out_dir="$1"
	shift
	"$PYTHON_BIN" -u "$ROOT_DIR/train_parallel.py" \
		"$@" \
		--out-dir "$out_dir"
}

has_saved_base_scores() {
	local out_dir="$1"
	local protocol="$2"
	compgen -G "$out_dir/_model_scores/$protocol/*.pkl" > /dev/null
}

echo "Running train_parallel.py"
echo "  python:       $PYTHON_BIN"
echo "  project_root: $PROJECT_ROOT"
echo "  results_root: $RESULTS_ROOT"
echo "  out_dir_base: $OUT_DIR_BASE"
echo "  base_scores:  $BASE_SCORES_ROOT"
echo "  protocols:    ${PROTOCOLS[*]}"
echo "  csv:          $CSV"
echo "  data_root:    $DATA_ROOT"
echo "  gpus:         $GPUS"
echo "  jobs/gpu:     $JOBS_PER_GPU"
echo "  early_stop:   enabled=$EARLY_STOPPING_ENABLED patience=$EARLY_STOPPING_PATIENCE"
echo "  val_ratio:    lpo=$effective_val_ratio_lpo lio=$effective_val_ratio_lio random=$effective_val_ratio_random"
echo "  dual_head:    $DUAL_HEAD_LIST"
echo "  head_2_scale: $HEAD_2_TARGET_SCALE_LIST"
echo "  head_2_loss:  $HEAD_2_LOSS_WEIGHT_LIST"
echo "  head_2_score: $HEAD_2_SCORE_WEIGHT_LIST"
echo "  sigmoid_log_s: $SIGMOID_LOG_S_LIST"
echo "  prior_scale:   $PRIOR_SCALE_LIST"
echo "  tail_scale:    $TAIL_SCALE_LIST"
echo "  lam_prior:     $LAM_PRIOR_LIST"
echo "  seeds:         $SEEDS"
echo "  skip_existing: $SKIP_EXISTING"

TOTAL_BASE_CONFIGS=$(( ${#PROTOCOLS[@]} * ${#RESOLUTIONS[@]} * ${#KS_VIEWS[@]} * ${#TARGET_SCALE_LIST[@]} * ${#HEAD_2_TARGET_SCALE_LIST[@]} * ${#SIGMOID_LOG_S_LIST[@]} * ${#DUAL_HEAD_LIST[@]} * ${#HEAD_2_LOSS_WEIGHT_LIST[@]} * ${#HEAD_2_SCORE_WEIGHT_LIST[@]} ))
TOTAL_SCORING_VARIANTS_PER_BASE=$(( ${#PRIOR_SCALE_LIST[@]} * ${#TAIL_SCALE_LIST[@]} * ${#LAM_PRIOR_LIST[@]} ))
echo "Total number of base train runs: $(( TOTAL_BASE_CONFIGS * SEED_COUNT ))"
echo "Total number of materialized seed runs: $(( TOTAL_BASE_CONFIGS * TOTAL_SCORING_VARIANTS_PER_BASE * SEED_COUNT ))"
echo "Total number of aggregated variants: $(( TOTAL_BASE_CONFIGS * TOTAL_SCORING_VARIANTS_PER_BASE ))"

for PROTOCOL in "${PROTOCOLS[@]}"; do
	for RESOLUTION in "${RESOLUTIONS[@]}"; do
		for K_VIEWS in "${KS_VIEWS[@]}"; do
			for target_scale in "${TARGET_SCALE_LIST[@]}"; do
				for head_2_target_scale in "${HEAD_2_TARGET_SCALE_LIST[@]}"; do
					for sigmoid_log_s in "${SIGMOID_LOG_S_LIST[@]}"; do
						for dual_head in "${DUAL_HEAD_LIST[@]}"; do
							for head_2_loss_weight in "${HEAD_2_LOSS_WEIGHT_LIST[@]}"; do
								for head_2_score_weight in "${HEAD_2_SCORE_WEIGHT_LIST[@]}"; do
									base_run_tag="scale${target_scale}_head2scale${head_2_target_scale}_sigmoidlogs${sigmoid_log_s}_dual${dual_head}_head2lw${head_2_loss_weight}_head2sw${head_2_score_weight}"
									base_run_dir_name="${base_run_tag}_res_${RESOLUTION}_k_views_${K_VIEWS}"
									base_protocol_out_dir="${BASE_SCORES_ROOT}/${PROTOCOL}/${base_run_dir_name}"

									shared_args=(
										--protocol "$PROTOCOL"
										--csv "$CSV"
										--data-root "$DATA_ROOT"
										--resolution "$RESOLUTION"
										--k-views "$K_VIEWS"
										--num-repetitions "$NUM_REPETITIONS"
										--batch-size "$BATCH_SIZE"
										--num-epochs "$NUM_EPOCHS"
										--lr "$LR"
										--weight-decay "$WEIGHT_DECAY"
										--num-workers "$NUM_WORKERS"
										--early-stopping-patience "$EARLY_STOPPING_PATIENCE"
										--test-ratio "$TEST_RATIO"
										--n-splits "$N_SPLITS"
										--gpus "$GPUS"
										--jobs-per-gpu "$JOBS_PER_GPU"
										--val-ratio-lpo "$effective_val_ratio_lpo"
										--val-ratio-lio "$effective_val_ratio_lio"
										--val-ratio-random "$effective_val_ratio_random"
										--target-scale "$target_scale"
										--head-2-target-scale "$head_2_target_scale"
										--head-2-loss-weight "$head_2_loss_weight"
										--head-2-score-weight "$head_2_score_weight"
										--sigmoid-log-s "$sigmoid_log_s"
									)

									if [[ "$CACHE_TRAIN" == "1" ]]; then shared_args+=(--cache-train); fi
									if [[ "$CACHE_TEST" == "1" ]]; then shared_args+=(--cache-test); fi
									if [[ -n "$MAX_PARALLEL" ]]; then shared_args+=(--max-parallel "$MAX_PARALLEL"); fi
									if [[ "$dual_head" == "1" ]]; then
										shared_args+=(--dual-head)
									else
										shared_args+=(--single-head)
									fi

									base_common_args=("${shared_args[@]}")
									if [[ -n "$TB_LOG_DIR" ]]; then
										base_common_args+=(--tb-log-dir "${TB_LOG_DIR}/${PROTOCOL}/${base_run_dir_name}")
										if [[ "$TB_LOG_VAL" == "1" ]]; then
											base_common_args+=(--tb-log-val)
										else
											base_common_args+=(--no-tb-log-val)
										fi
									fi

									for seed in "${SEED_LIST[@]}"; do
										base_seed_out_dir="${base_protocol_out_dir}/seed${seed}"
										if has_saved_base_scores "$base_seed_out_dir" "$PROTOCOL"; then
											echo "Skipping existing base scores: ${base_seed_out_dir}"
											continue
										fi

										mkdir -p "$base_seed_out_dir"
										echo "Training base scores: seed=$seed target_scale=$target_scale head_2_target_scale=$head_2_target_scale sigmoid_log_s=$sigmoid_log_s dual_head=$dual_head head_2_loss_weight=$head_2_loss_weight head_2_score_weight=$head_2_score_weight"
										run_train_parallel "$base_seed_out_dir" "${base_common_args[@]}" --seed "$seed" --base-scores-only
									done

									for prior_scale in "${PRIOR_SCALE_LIST[@]}"; do
										for tail_scale in "${TAIL_SCALE_LIST[@]}"; do
											for lam_prior in "${LAM_PRIOR_LIST[@]}"; do
												variant_run_tag="${base_run_tag}_priorscale${prior_scale}_lamprior${lam_prior}_tailscale${tail_scale}"
												variant_run_dir_name="${variant_run_tag}_res_${RESOLUTION}_k_views_${K_VIEWS}"
												variant_protocol_out_dir="${OUT_DIR_BASE}/${PROTOCOL}/${variant_run_dir_name}"
												result_suffix="priorscale_${prior_scale}_sigmoidlogs_${sigmoid_log_s}_tailscale_${tail_scale}_head2lossweight_${head_2_loss_weight}_head2scoreweight_${head_2_score_weight}_head2targetscale_${head_2_target_scale}_lamprior_${lam_prior}_res_${RESOLUTION}_k_views_${K_VIEWS}"

												variant_common_args=(
													"${shared_args[@]}"
													--prior-scale "$prior_scale"
													--tail-scale "$tail_scale"
													--lam-prior "$lam_prior"
												)
												if [[ -n "$TB_LOG_DIR" ]]; then
													variant_common_args+=(--tb-log-dir "${TB_LOG_DIR}/${PROTOCOL}/${variant_run_dir_name}")
													if [[ "$TB_LOG_VAL" == "1" ]]; then
														variant_common_args+=(--tb-log-val)
													else
														variant_common_args+=(--no-tb-log-val)
													fi
												fi

												aggregate_seed_dirs=()
												for seed in "${SEED_LIST[@]}"; do
													base_seed_out_dir="${base_protocol_out_dir}/seed${seed}"
													run_out_dir="${variant_protocol_out_dir}/seed${seed}"
													aggregate_seed_dirs+=("$run_out_dir")
													final_csv="${run_out_dir}/res_${result_suffix}.csv"

													if [[ "$SKIP_EXISTING" == "1" && -f "$final_csv" ]]; then
														echo "Skipping existing seed config: $final_csv"
														continue
													fi

													mkdir -p "$run_out_dir"
													echo "Materializing: seed=$seed target_scale=$target_scale head_2_target_scale=$head_2_target_scale sigmoid_log_s=$sigmoid_log_s prior_scale=$prior_scale tail_scale=$tail_scale lam_prior=$lam_prior dual_head=$dual_head head_2_loss_weight=$head_2_loss_weight head_2_score_weight=$head_2_score_weight"
													run_train_parallel "$run_out_dir" "${variant_common_args[@]}" --seed "$seed" --materialize-only --base-scores-dir "$base_seed_out_dir"
												done

												if (( SEED_COUNT > 1 )); then
													aggregate_seed_dirs_csv="$(join_by_comma "${aggregate_seed_dirs[@]}")"
													echo "Aggregating seeds into: $variant_protocol_out_dir"
													run_train_parallel "$variant_protocol_out_dir" "${variant_common_args[@]}" --aggregate-seed-dirs "$aggregate_seed_dirs_csv"
												fi
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
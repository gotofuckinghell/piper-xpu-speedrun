#!/usr/bin/env bash
# ABBA-проверка foreach AdamW: 4 блока по 20 шагов, чередование on/off/off/on.
set -u
S="<WORK>"
V33="<PROJECT>/datasets/piper_yt/dataset_v33"
PY="<HOME>/.venvs/piper-main/Scripts/python.exe"
export OMP_NUM_THREADS=30 MKL_NUM_THREADS=30 OPENBLAS_NUM_THREADS=30 NUMEXPR_NUM_THREADS=30
export PIPER_XPU=1 PIPER_NO_MOS=1 PIPER_NO_AUDIO_LOG=1 PIPER_AMP=bf16 PIPER_CKPT=1

run_block() {
  local arm="$1"
  local round="$2"
  local fe="$3"
  local tag="abba_${arm}_${round}"
  export PIPER_FOREACH_ADAM="$fe"
  "$PY" -m piper.train fit \
    --data.voice_name "$tag" \
    --data.csv_path "$V33/metadata_nohdr.csv" --data.audio_dir "$V33/wavs" \
    --model.sample_rate 22050 --data.espeak_voice ru \
    --data.cache_dir "$S/cache_v33" --data.config_path "$S/mm_v13_config.json" \
    --data.batch_size 112 --data.num_workers 0 \
    --data.validation_split 0.02 --data.length_bucketing true \
    --data.bucket_multiplier 16 --model.segment_size 2048 \
    --model.p_dropout 0.10 \
    --model.warmstart_ckpt "<PROJECT>/voice/<VOICE_DIR>/ckpt/v30_slim.ckpt" \
    --trainer.max_steps 20 --trainer.max_epochs 40 \
    --trainer.val_check_interval 10000 --trainer.limit_val_batches 0 \
    --trainer.num_sanity_val_steps 0 --trainer.log_every_n_steps 20 \
    --trainer.default_root_dir "$S/run_$tag" \
    --trainer.enable_progress_bar false --trainer.enable_model_summary false \
    > "$S/$tag.log" 2>&1
}

for spec in "on 1 1" "off 1 0" "off 2 0" "on 2 1"; do
  set -- $spec
  run_block "$1" "$2" "$3"
  rate=$(rg -N "PIPER_TIMING steps=20 seconds=" "$S/abba_$1_$2.log" | tail -1 | sed 's/.*seconds=//')
  echo "{\"arm\": \"$1\", \"round\": $2, \"seconds\": $rate}" >> "$S/foreach_abba.jsonl"
done
rg -N "seconds=" "$S"/abba_*.log | tail -4

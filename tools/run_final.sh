#!/usr/bin/env bash
# Финальный прогон: bf16 + checkpoints, 112x2048, корпус v33 (25,1 ч).
# Проверенный конфиг: 26,8 реплики/с, эпоха ~10 мин, GPU 45,5%.
set -u
S="<WORK>"
V33="<PROJECT>/datasets/piper_yt/dataset_v33"
CKPT="<PROJECT>/voice/<VOICE_DIR>/ckpt"
PY="<HOME>/.venvs/piper-main/Scripts/python.exe"

export OMP_NUM_THREADS=30 MKL_NUM_THREADS=30 OPENBLAS_NUM_THREADS=30 NUMEXPR_NUM_THREADS=30
export PIPER_XPU=1 PIPER_NO_MOS=1 PIPER_NO_AUDIO_LOG=1
export PIPER_AMP=bf16 PIPER_CKPT=1
export PIPER_LR="${PIPER_LR:-1e-5}" PIPER_LR_D="${PIPER_LR_D:-1e-5}"

TAG="v34_final"
rm -rf "$S/run_$TAG"

"$PY" -m piper.train fit \
  --data.voice_name "$TAG" \
  --data.csv_path "$V33/metadata_nohdr.csv" \
  --data.audio_dir "$V33/wavs" \
  --model.sample_rate 22050 --data.espeak_voice ru \
  --data.cache_dir "$S/cache_v33" \
  --data.config_path "$S/mm_v13_config.json" \
  --data.batch_size 112 --data.num_workers 0 \
  --data.validation_split 0.02 --data.length_bucketing true \
  --data.bucket_multiplier 16 \
  --model.segment_size 2048 --model.p_dropout 0.10 \
  --model.warmstart_ckpt "$CKPT/v30_slim.ckpt" \
  --trainer.max_steps "${MAX_STEPS:-6000}" \
  --trainer.max_epochs 40 \
  --trainer.val_check_interval 2000 --trainer.limit_val_batches 20 \
  --trainer.num_sanity_val_steps 0 \
  --trainer.log_every_n_steps 50 \
  --trainer.default_root_dir "$S/run_$TAG" \
  --trainer.enable_progress_bar false --trainer.enable_model_summary false

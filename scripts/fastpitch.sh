#!/usr/bin/env bash
# nohup bash fastpitch.sh > log.out 2>&1 &

: ${FASTPVOL:="/s/chopin/l/grad/stock/nvme/data/cs542/project/DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch"}
: ${DVOL:="/s/chopin/l/grad/stock/nvme/data/cs542/project/tweeteval/datasets"}
: ${DATASET:="train"}
: ${TASK:="sentiment"}

: ${PHRASES:="$DVOL/audio/phrases/${TASK}/${DATASET}.tsv"}
: ${OUTPUT_DIR:="$DVOL/audio/output/${TASK}/${DATASET}"}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}

echo -e "\nPHRASES=$PHRASES, OUTPUT_DIR=$OUTPUT_DIR\n"

# : ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${WAVEGLOW:="SKIP"}
: ${FASTPITCH:="pretrained_models/fastpitch/nvidia_fastpitch_210824.pt"}

: ${BATCH_SIZE:=32}

: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${PACE:=1.2}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}
: ${SAVE_MELS:=true}

: ${SPEAKER:=0}
: ${NUM_SPEAKERS:=1}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" -i $PHRASES"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"

# only use --wn-channels if WAVEGLOW is not 'SKIP'
ARGS+=" --waveglow $WAVEGLOW"
# ARGS+=" --wn-channels 256"

ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --text-cleaners english_cleaners_v2"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --pace $PACE"
ARGS+=" --repeats $REPEATS"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --speaker $SPEAKER"
ARGS+=" --n-speakers $NUM_SPEAKERS"
[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"
[ "$SAVE_MELS" = "true" ]   && ARGS+=" --save-mels"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
pushd $FASTPVOL
time python inference.py $ARGS "$@"
popd

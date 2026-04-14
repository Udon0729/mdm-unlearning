#!/bin/bash
# cd to project root before running this script
# Use the active Python environment
CKPT=${MDM_CKPT:-workdir/untrac/mdm-untrac-113M-40000steps/iter-040000-ckpt.pth}
OUTDIR=workdir/untrac/reconstruction_eu
LOGDIR=workdir/untrac/reconstruction_eu/logs
mkdir -p $OUTDIR $LOGDIR

# EU uses 10000 steps (same as KL comparison)

run_recon() {
    local corpus=$1
    local gpu=$2
    echo "[$(date +%H:%M:%S)] Start: corpus=$corpus GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mdm_unlearning.evaluate.reconstruction_mdm \
        --model 113 --seq_len 1024 \
        --ckpt_path $CKPT --data_dir data/untrac \
        --unlearn_corpus $corpus --unlearn_steps 10000 \
        --unlearn_method eu --eu_lambda 1.0 \
        --num_samples 100 --mask_ratio 0.15 \
        --output $OUTDIR/recon_${corpus}.json \
        &> $LOGDIR/recon_${corpus}.log
    echo "[$(date +%H:%M:%S)] Done: corpus=$corpus (exit=$?)"
}

# GPU 0 in use
run_recon bookcorpus 1 &
run_recon stackexchange 2 &
run_recon ccnewsv2 3 &
run_recon gutenberg 4 &
run_recon hackernews 5 &
run_recon openwebtext 6 &
wait
echo "[$(date +%H:%M:%S)] First 6 done"

run_recon pilecc 1 &
run_recon wikipedia 2 &
wait
echo "[$(date +%H:%M:%S)] ALL DONE"

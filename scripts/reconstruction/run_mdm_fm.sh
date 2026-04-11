#!/bin/bash
# cd to project root before running this script
# Use the active Python environment
CKPT=${MDM_CKPT:-workdir/untrac/mdm-untrac-113M-40000steps/iter-040000-ckpt.pth}
OUTDIR=workdir/untrac/reconstruction_fm
LOGDIR=workdir/untrac/reconstruction_fm/logs
mkdir -p $OUTDIR $LOGDIR

# Fisher-Meta: best step was 2000
run_recon() {
    local corpus=$1
    local gpu=$2
    echo "[$(date +%H:%M:%S)] Start: corpus=$corpus GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mdm_unlearning.evaluate.reconstruction_mdm \
        --model 113 --seq_len 1024 \
        --ckpt_path $CKPT --data_dir data/untrac \
        --unlearn_corpus $corpus --unlearn_steps 2000 \
        --unlearn_method fisher_meta \
        --ewc_alpha 1.0 --saliency_top_pct 30 --fisher_bottom_pct 70 \
        --fisher_samples 500 --meta_k 10 --meta_every 500 --meta_beta 0.01 \
        --num_samples 100 --mask_ratio 0.15 \
        --output $OUTDIR/recon_${corpus}.json \
        &> $LOGDIR/recon_${corpus}.log
    echo "[$(date +%H:%M:%S)] Done: corpus=$corpus (exit=$?)"
}

run_recon bookcorpus 0 &
run_recon stackexchange 1 &
run_recon ccnewsv2 2 &
run_recon gutenberg 3 &
run_recon hackernews 4 &
run_recon openwebtext 5 &
run_recon pilecc 6 &
wait
echo "[$(date +%H:%M:%S)] First 7 done"

run_recon wikipedia 0 &
wait
echo "[$(date +%H:%M:%S)] ALL DONE"

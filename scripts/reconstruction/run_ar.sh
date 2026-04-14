#!/bin/bash
# cd to project root before running this script
# Use the active Python environment
CKPT=${AR_CKPT:-workdir/untrac/ar-untrac-113M-40000steps/iter-040000-ckpt.pth}
OUTDIR=workdir/untrac/reconstruction_ar
LOGDIR=workdir/untrac/reconstruction_ar/logs
mkdir -p $OUTDIR $LOGDIR

run_recon() {
    local corpus=$1
    local gpu=$2
    local outfile=$OUTDIR/recon_${corpus}.json
    if [ -f "$outfile" ]; then echo "Skip: $corpus (exists)"; return; fi
    echo "[$(date +%H:%M:%S)] Start: corpus=$corpus GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mdm_unlearning.evaluate.reconstruction_ar \
        --model 113 --seq_len 1024 --ckpt_path $CKPT --data_dir data/untrac \
        --unlearn_corpus $corpus --unlearn_steps 10000 --eu_lambda 1.0 \
        --num_samples 100 --output $outfile &> $LOGDIR/recon_${corpus}.log
    echo "[$(date +%H:%M:%S)] Done: corpus=$corpus"
}

# GPU 1,2 available, 2 at a time
for pair in "stackexchange hackernews" "ccnewsv2 openwebtext" "pilecc wikipedia"; do
    c1=$(echo $pair | cut -d' ' -f1)
    c2=$(echo $pair | cut -d' ' -f2)
    run_recon $c1 1 &
    run_recon $c2 2 &
    wait
done
echo "[$(date +%H:%M:%S)] ALL ARM RECONSTRUCTION COMPLETE"

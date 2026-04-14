#!/bin/bash
# cd to project root before running this script
# Use the active Python environment
CKPT=${AR_CKPT:-workdir/untrac/ar-untrac-113M-40000steps/iter-040000-ckpt.pth}
LOGDIR=workdir/untrac/ar_results/logs
mkdir -p workdir/untrac/ar_results/kl workdir/untrac/ar_results/eu $LOGDIR

# GPU 0-2 occupied. Use GPU 3-6

run_untrac() {
    local method=$1
    local corpus=$2
    local gpu=$3
    local outdir=workdir/untrac/ar_results/$method
    local logfile=$LOGDIR/${method}_${corpus}.log
    echo "[$(date +%H:%M:%S)] Start: method=$method corpus=$corpus GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mdm_unlearning.evaluate.untrac_ar \
        --mode untrac --model 113 --seq_len 1024 \
        --ckpt_path $CKPT --test_dataset all --data_dir data/untrac \
        --mc_num 1 --mc_batch 32 \
        --unlearn_lr 5e-5 --unlearn_epochs 1 --unlearn_batch_size 1 \
        --eval_steps 2000 \
        --unlearn_method $method --kl_alpha 1.0 --eu_lambda 1.0 \
        --untrac_corpus $corpus \
        --output $outdir/${method}_${corpus}.json \
        &> $logfile
    echo "[$(date +%H:%M:%S)] Done: method=$method corpus=$corpus (exit=$?)"
}

# KL method: 4 corpora in parallel on GPU 3-6, then 4 more
for method in kl eu; do
    echo "=== Running $method ==="
    run_untrac $method bookcorpus 3 &
    run_untrac $method stackexchange 4 &
    run_untrac $method ccnewsv2 5 &
    run_untrac $method gutenberg 6 &
    wait
    run_untrac $method hackernews 3 &
    run_untrac $method openwebtext 4 &
    run_untrac $method pilecc 5 &
    run_untrac $method wikipedia 6 &
    wait
    echo "=== $method complete ==="
done
echo "[$(date +%H:%M:%S)] ALL ARM UNTRAC COMPLETE"

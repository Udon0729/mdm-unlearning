#!/bin/bash
# cd to project root before running this script
# Use the active Python environment
CKPT=${E2D2_CKPT:-workdir/untrac/e2d2-untrac-113M-40000steps/iter-040000-ckpt.pth}
OUTDIR=workdir/untrac/e2d2_kl
LOGDIR=workdir/untrac/e2d2_kl/logs
mkdir -p $OUTDIR $LOGDIR

python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"

run_untrac() {
    local corpus=$1
    local gpu=$2
    local logfile=$LOGDIR/kl_${corpus}.log
    echo "[$(date +%H:%M:%S)] Start: corpus=$corpus GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mdm_unlearning.evaluate.untrac_e2d2 \
        --mode untrac --model 113 --seq_len 1024 \
        --ckpt_path $CKPT --test_dataset all --data_dir data/untrac \
        --mc_num 1 --mc_batch 32 \
        --unlearn_lr 5e-5 --unlearn_epochs 1 --unlearn_batch_size 1 \
        --eval_steps 2000 \
        --unlearn_method kl --kl_alpha 1.0 \
        --untrac_corpus $corpus \
        --output $OUTDIR/kl_${corpus}.json \
        &> $logfile
    echo "[$(date +%H:%M:%S)] Done: corpus=$corpus (exit=$?)"
}

# GPU 0 is in use. Use GPUs 1-6 (6 GPUs)
run_untrac bookcorpus 1 &
run_untrac stackexchange 2 &
run_untrac ccnewsv2 3 &
run_untrac gutenberg 4 &
run_untrac hackernews 5 &
run_untrac openwebtext 6 &
wait
echo "[$(date +%H:%M:%S)] First 6 done"

run_untrac pilecc 1 &
run_untrac wikipedia 2 &
wait
echo "[$(date +%H:%M:%S)] ALL E2D2 UNTRAC COMPLETE"

#!/bin/bash
# cd to project root before running this script
# Use the active Python environment
CKPT=${MDM_CKPT:-workdir/untrac/mdm-untrac-113M-40000steps/iter-040000-ckpt.pth}
OUTDIR=workdir/untrac/eu
LOGDIR=workdir/untrac/eu/logs
mkdir -p $OUTDIR $LOGDIR

python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"

run_eu() {
    local corpus=$1
    local gpu=$2
    local logfile=$LOGDIR/eu_${corpus}.log
    echo "[$(date +%H:%M:%S)] Start: corpus=$corpus GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mdm_unlearning.evaluate.untrac_mdm \
        --mode untrac --model 113 --seq_len 1024 \
        --ckpt_path $CKPT --test_dataset all --data_dir data/untrac \
        --mc_num 1 --mc_batch 32 \
        --unlearn_lr 5e-5 --unlearn_epochs 1 --unlearn_batch_size 1 \
        --eval_steps 2000 \
        --unlearn_method eu --eu_lambda 1.0 \
        --untrac_corpus $corpus \
        --output $OUTDIR/eu_${corpus}.json \
        &> $logfile
    echo "[$(date +%H:%M:%S)] Done: corpus=$corpus (exit=$?)"
}

# GPU 0 in use. Use GPUs 1-6
run_eu bookcorpus 1 &
run_eu stackexchange 2 &
run_eu ccnewsv2 3 &
run_eu gutenberg 4 &
run_eu hackernews 5 &
run_eu openwebtext 6 &
wait
echo "[$(date +%H:%M:%S)] First 6 done"

run_eu pilecc 1 &
run_eu wikipedia 2 &
wait
echo "[$(date +%H:%M:%S)] ALL EU COMPLETE"

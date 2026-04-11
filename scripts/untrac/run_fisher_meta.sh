#!/bin/bash
# cd to project root before running this script
# Use the active Python environment
CKPT=${MDM_CKPT:-workdir/untrac/mdm-untrac-113M-40000steps/iter-040000-ckpt.pth}
OUTDIR=workdir/untrac/fisher_meta
LOGDIR=workdir/untrac/fisher_meta/logs
mkdir -p $OUTDIR $LOGDIR

python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"

run_fm() {
    local corpus=$1
    local gpu=$2
    local logfile=$LOGDIR/fm_${corpus}.log
    echo "[$(date +%H:%M:%S)] Start: corpus=$corpus GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mdm_unlearning.evaluate.untrac_mdm \
        --mode untrac --model 113 --seq_len 1024 \
        --ckpt_path $CKPT --test_dataset all --data_dir data/untrac \
        --mc_num 1 --mc_batch 32 \
        --unlearn_lr 5e-5 --unlearn_epochs 1 --unlearn_batch_size 1 \
        --eval_steps 2000 \
        --unlearn_method fisher_meta \
        --ewc_alpha 1.0 \
        --saliency_top_pct 30 --fisher_bottom_pct 70 \
        --fisher_samples 500 \
        --meta_k 10 --meta_every 500 --meta_beta 0.01 \
        --untrac_corpus $corpus \
        --output $OUTDIR/fm_${corpus}.json \
        &> $logfile
    echo "[$(date +%H:%M:%S)] Done: corpus=$corpus (exit=$?)"
}

run_fm bookcorpus 0 &
run_fm stackexchange 1 &
run_fm ccnewsv2 2 &
run_fm gutenberg 3 &
run_fm hackernews 4 &
run_fm openwebtext 5 &
run_fm pilecc 6 &
wait
echo "[$(date +%H:%M:%S)] First 7 done"

run_fm wikipedia 0 &
wait
echo "[$(date +%H:%M:%S)] ALL FISHER_META COMPLETE"

#!/bin/bash
set -euo pipefail
# cd to project root before running this script
# Use the active Python environment
CKPT=${MDM_CKPT:-workdir/untrac/mdm-untrac-113M-40000steps/iter-040000-ckpt.pth}
LOGDIR=workdir/untrac/logs
mkdir -p $LOGDIR

# Verify CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA OK: {torch.cuda.device_count()} GPUs')"

run_untrac() {
    local corpus=$1
    local gpu=$2
    local logfile=$LOGDIR/untrac_${corpus}.log
    echo "[$(date +%H:%M:%S)] UnTrac start: corpus=$corpus GPU=$gpu log=$logfile"
    CUDA_VISIBLE_DEVICES=$gpu python -m mdm_unlearning.evaluate.untrac_mdm \
        --mode untrac --model 113 --seq_len 1024 \
        --ckpt_path $CKPT --test_dataset all --data_dir data/untrac \
        --mc_num 1 --mc_batch 32 \
        --unlearn_lr 5e-5 --unlearn_epochs 1 --unlearn_batch_size 1 \
        --eval_steps 500 --untrac_corpus $corpus \
        --output workdir/untrac/untrac_${corpus}.json &> $logfile
    echo "[$(date +%H:%M:%S)] UnTrac done: corpus=$corpus (exit=$?)"
}

# Run 7 in parallel, then 1
run_untrac bookcorpus 0 &
run_untrac stackexchange 1 &
run_untrac ccnewsv2 2 &
run_untrac gutenberg 3 &
run_untrac hackernews 4 &
run_untrac openwebtext 5 &
run_untrac pilecc 6 &
wait
echo "[$(date +%H:%M:%S)] First 7 done"

run_untrac wikipedia 0 &
wait
echo "[$(date +%H:%M:%S)] ALL UNTRAC COMPLETE"

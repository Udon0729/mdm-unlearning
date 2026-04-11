#!/bin/bash
# cd to project root before running this script
# Use the active Python environment
CORPORA=(bookcorpus stackexchange ccnewsv2 gutenberg hackernews openwebtext pilecc wikipedia)
GPUS=(4 5 6)

for idx in "${!CORPORA[@]}"; do
    corpus=${CORPORA[$idx]}
    gpu=${GPUS[$((idx % 3))]}
    echo "[$(date +%H:%M:%S)] LOO $((idx+1))/8: exclude=$corpus GPU=$gpu"
    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu \
        python -m mdm_unlearning.train.train_mdm \
        --model 113 --max_steps 40000 --data_dir data/untrac \
        --save_interval 40000 --num_devices 1 --batch_size 8 --micro_batch_size 4 \
        --grad_clip 1.0 --seq_len 1024 --exclude_corpus $corpus 2>&1 | tail -3
    echo "[$(date +%H:%M:%S)] Done: $corpus"
done
echo "[$(date +%H:%M:%S)] ALL LOO COMPLETE"

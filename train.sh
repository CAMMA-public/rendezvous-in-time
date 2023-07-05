#! /bin/bash

CUDA_VISIBLE_DEVICES=1, python3 main.py \
    --exp_name rit_train \
    --log_name rit_train.log \
    --max_epochs 40 \
    --early_stopping_patience 5 \
    --fold 1 \
    --split 'cholect45-crossval' \
    --m 3 \
    --bs 16 \
    --nw 6 \
    --topK 5 \
    --ln 0 \
    --cg 0 \
    --od1 1e-6 \
    --od2 1e-6 \
    --od3 1e-6 \
    --mom 0.95 \
    --ms1 20 \
    --ms2 39 \
    --ms3 60 \
    --g1 0.94 \
    --g2 0.95 \
    --g3 0.99 \
    --layers 8 \
    --data_dir "path-to-CholecT50" \
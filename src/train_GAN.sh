#!/bin/bash
iteration=$1
set -e
prefix=${bnd_type}_iter${iteration}_${setting}_gan

# Train GAN and output phoneme posterior
cd GAN-based-model

python3 main.py --mode train --cuda_id $CUDA_VISIBLE_DEVICES \
               --bnd_type $bnd_type --iteration $iteration \
               --setting $setting \
               --data_dir $DATA_PATH \
               --save_dir $DATA_PATH/save/${prefix} \
               --config "./config.yaml"
exit 1
cd ../ 

# WFST decode the phoneme sequences
cd WFST-decoder
python3 scripts/decode.py --set_type test --lm_type $setting \
                         --data_path $DATA_PATH --prefix $prefix \
                         --jobs $jobs
python3 scripts/decode.py --set_type train --lm_type $setting \
                         --data_path $DATA_PATH --prefix $prefix \
                         --jobs $jobs
cd ../

# Evalution
python3 eval_per.py --bnd_type $bnd_type --set_type test --lm_type $setting \
                   --data_path $DATA_PATH --prefix $prefix \
                   --file_name test_output.txt | tee $DATA_PATH/result/${prefix}.log
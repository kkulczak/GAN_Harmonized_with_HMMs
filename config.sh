# !/bin/bash

#Path
export ROOT_DIR=/home/shared #abs. path of this github repository
export TIMIT_DIR=/home/shared/data/timit_data #abs. path of your timit dataset
export DATA_PATH=${ROOT_DIR}/data
export TF_RANDOM_SEED=0
#Boundaries type: orc / uns
export bnd_type=orc

#Setting: match / nonmatch
export setting=match

#Number of jobs
export jobs=12

#Use tensorflow debugger: on / off
export TFGDB=off
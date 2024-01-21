#!bin/sh
CONFIG=$1
WORKDIR=$2

shift 2

python train_w_nerf_binary_occ.py --py-config $CONFIG --work-dir $WORKDIR "$@"

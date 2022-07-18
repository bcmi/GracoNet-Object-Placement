#!/bin/bash

### Calculating FID (plausibility) ###

### START USAGE ###
# sh script/eval_fid.sh ${EXPID} ${EPOCH} ${FID_GT_IMGS}
### END USAGE ###

EXPID=$1
EPOCH=$2
FID_GT_IMGS=$3

python eval/fid_resize299.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"
python eval/fid_score.py result/${EXPID}/eval/${EPOCH}/images299/ ${FID_GT_IMGS} --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"

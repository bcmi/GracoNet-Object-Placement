#!/bin/bash

### Calculating SimOPA Accuracy (plausibility) ###

### START USAGE ###
# sh script/eval_acc.sh ${EXPID} ${EPOCH} ${SIMOPA_MODEL}
### END USAGE ###

EXPID=$1
EPOCH=$2
SIMOPA_MODEL=$3

cd faster-rcnn
python generate_tsv.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval" --cuda
python convert_data.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"
cd ..
python eval/simopa_acc.py --checkpoint ${SIMOPA_MODEL} --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"

### Note off the following lines if you would like to delete faster-rcnn intermediate results ###
# rm result/${EXPID}/eval/${EPOCH}/eval_roiinfos.csv
# rm result/${EXPID}/eval/${EPOCH}/eval_fgfeats.npy
# rm result/${EXPID}/eval/${EPOCH}/eval_scores.npy
# rm result/${EXPID}/eval/${EPOCH}/eval_feats.npy
# rm result/${EXPID}/eval/${EPOCH}/eval_bboxes.npy

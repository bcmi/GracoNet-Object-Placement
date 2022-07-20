#!/bin/bash

### Calculating Accuracy (plausibility) ###

### START USAGE ###
# sh script/eval_acc.sh ${EXPID} ${EPOCH} ${BINARY_CLASSIFIER}
### END USAGE ###

EXPID=$1
EPOCH=$2
BINARY_CLASSIFIER=$3

cd faster-rcnn
python generate_tsv.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval" --cuda
python convert_data.py --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"
cd ..
python eval/simopa_acc.py --checkpoint ${BINARY_CLASSIFIER} --expid ${EXPID} --epoch ${EPOCH} --eval_type "eval"

### Uncomment the following lines if you would like to delete faster-rcnn intermediate results ###
# rm result/${EXPID}/eval/${EPOCH}/eval_roiinfos.csv
# rm result/${EXPID}/eval/${EPOCH}/eval_fgfeats.npy
# rm result/${EXPID}/eval/${EPOCH}/eval_scores.npy
# rm result/${EXPID}/eval/${EPOCH}/eval_feats.npy
# rm result/${EXPID}/eval/${EPOCH}/eval_bboxes.npy

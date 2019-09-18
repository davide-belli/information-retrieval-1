#!/bin/bash

VALIDATION_OUT='./evaluations'
VALIDATION_IN='./ap_88_89/qrel_validation'
TEST_OUT='./test'
TEST_IN='./ap_88_89/qrel_test'
INDIR='./rankings'

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--indir)
    INDIR="$2"
    shift
    ;;
esac
shift
done

mkdir -p $VALIDATION_OUT
mkdir -p $TEST_OUT

measures=('ndcg_cut_10' 'map_cut_1000' 'P_5' 'recall_1000')

regex_measures=$(printf "|(%s)" "${measures[@]}")
regex_measures=${regex_measures:1}

echo $regex_measures

for m in $INDIR/*.run
do
    model=$(basename $m)
    model="${model%.*}"
    echo $model
    for meas in ${measures[@]}; do    
        printf "\t$meas\n"
        trec_eval -m all_trec -q $VALIDATION_IN $m | grep -E "^$meas\s" > $VALIDATION_OUT/$model.$meas.eval
        trec_eval -m all_trec -q $TEST_IN $m | grep -E "^$meas\s" > $TEST_OUT/$model.$meas.eval
    done
    printf "\tall\n"
    trec_eval -m all_trec -q $VALIDATION_IN $m | grep -E "\sall\s" | grep -E "^($regex_measures)\s" > $VALIDATION_OUT/$model.all
    trec_eval -m all_trec -q $TEST_IN $m | grep -E "\sall\s" | grep -E "^($regex_measures)\s" > $TEST_OUT/$model.all
done    



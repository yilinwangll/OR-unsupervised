MODEL_PATH=$1
NUM_GPUS=$2

sh eval_legacy/eval.NL4OPT.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval_legacy/eval.MAMO_EasyLP.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval_legacy/eval.MAMO_ComplexLP.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval_legacy/eval.IndustryOR.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval_legacy/eval.NLP4LP.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval_legacy/eval.ComplexOR.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval_legacy/eval.OptiBench.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval_legacy/eval.ICMLTEST.pass1.sh $MODEL_PATH $NUM_GPUS

# sh eval_legacy/eval.NL4OPT.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval_legacy/eval.MAMO_EasyLP.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval_legacy/eval.MAMO_ComplexLP.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval_legacy/eval.IndustryOR.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval_legacy/eval.NLP4LP.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval_legacy/eval.ComplexOR.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval_legacy/eval.OptiBench.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval_legacy/eval.ICMLTEST.pass8.sh $MODEL_PATH $NUM_GPUS

python eval_legacy/read_eval_results.py $MODEL_PATH

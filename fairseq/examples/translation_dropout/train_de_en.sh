#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
#BPE_TOKENS=10000
SRC_BPE_TOKENS=10000
TGT_BPE_TOKENS=10000
SRC_DROPOUT=0.0
TGT_DROPOUT=0.0
SEED=0
DEVICE=0

EXPERIMENT_PREFIX="experiment"

while [[ "$#" -gt 0 ]]
do case $1 in
    --src-bpe-tokens) SRC_BPE_TOKENS=$2
    shift;;
    --tgt-bpe-tokens) TGT_BPE_TOKENS=$2
    shift;;
    --src-dropout) SRC_DROPOUT=$2
    shift;;
    --tgt-dropout) TGT_DROPOUT=$2
    shift;;
    --seed) SEED=$2
    shift;;
    --device) DEVICE=$2
    shift;;
    --experiment-name) EXPERIMENT_PREFIX="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done
echo "========= PARAMETERS =========== "
echo -e "SRC_TOKENS $SRC_BPE_TOKENS \nTGT_TOKENS $TGT_BPE_TOKENS \nSRC_DROPOUT $SRC_DROPOUT \nTGT_DROPOUT $TGT_DROPOUT \nSEED $SEED \nDEVICE $DEVICE \nNAME $EXPERIMENT_PREFIX\n"
echo "========= PARAMETERS =========== "


src=de
tgt=en
lang=de-en

EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_BPE_${SRC_BPE_TOKENS}_${TGT_BPE_TOKENS}_dropout_${SRC_DROPOUT}_${TGT_DROPOUT}_seed_${SEED}.${lang}"

prep=experiments/$EXPERIMENT_NAME
tmp=$prep/tmp
orig=orig


if [ -d "../../experiment_outputs/${EXPERIMENT_NAME}" ]
then
    echo "${EXPERIMENT_NAME} already done, SKIPPING"
    exit 0
fi

mkdir $prep

mkdir data-bin/$EXPERIMENT_NAME

BPE_CODE=$prep/code
BPE_VOCAB=$prep/vocab

echo "learn_BPE for src: $src"
python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $orig/train.$src -s $SRC_BPE_TOKENS -t -o $BPE_CODE.$src --write-vocabulary $BPE_VOCAB.$src

echo "learn_BPE for tgt: $tgt"
python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $orig/train.$tgt -s $TGT_BPE_TOKENS -t -o $BPE_CODE.$tgt --write-vocabulary $BPE_VOCAB.$tgt


for f in train valid test; do
    echo "apply_bpe.py ($src) to ${f}.${src}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE.$src --vocabulary $BPE_VOCAB.$src < $orig/$f.$src > $prep/$f.$src
    cp $prep/$f.$src data-bin/$EXPERIMENT_NAME/$f.$src

    echo "apply_bpe.py ($tgt) to ${f}.${tgt}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE.$tgt --vocabulary $BPE_VOCAB.$tgt < $orig/$f.$tgt > $prep/$f.$tgt
    cp $prep/$f.$tgt data-bin/$EXPERIMENT_NAME/$f.$tgt
done

cd ../..

TEXT=examples/translation_dropout/experiments/$EXPERIMENT_NAME
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir examples/translation_dropout/data-bin/$EXPERIMENT_NAME \
    --workers 8 \
    --srcdict examples/translation_dropout/experiments/$EXPERIMENT_NAME/vocab.$src \
    --tgtdict examples/translation_dropout/experiments/$EXPERIMENT_NAME/vocab.$tgt

# cp $orig/train.$src examples/translation_dropout/data-bin/$EXPERIMENT_NAME/train.raw.$src
# cp $orig/train.$tgt examples/translation_dropout/data-bin/$EXPERIMENT_NAME/train.raw.$tgt

cp ${TEXT}/vocab.$src ${TEXT}/vocab.$tgt examples/translation_dropout/data-bin/$EXPERIMENT_NAME/
cp ${TEXT}/code.$src ${TEXT}/code.$tgt examples/translation_dropout/data-bin/$EXPERIMENT_NAME/

# sed -i -r 's/(@@ )|(@@ ?$)//g' examples/translation_dropout/data-bin/$EXPERIMENT_NAME/train.raw.$src
# sed -i -r 's/(@@ )|(@@ ?$)//g' examples/translation_dropout/data-bin/$EXPERIMENT_NAME/train.raw.$tgt

mkdir experiment_outputs/${EXPERIMENT_NAME}/

CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-train  examples/translation_dropout/data-bin/$EXPERIMENT_NAME \
                                            --arch transformer_iwslt_de_en \
                                            --share-decoder-input-output-embed \
                                            --optimizer adam --adam-betas '(0.9, 0.98)' \
                                            --clip-norm 0.0 \
                                            --lr 5e-4 \
                                            --lr-scheduler inverse_sqrt \
                                            --warmup-updates 4000 \
                                            --dropout 0.3 \
                                            --weight-decay 0.0001 \
                                            --criterion label_smoothed_cross_entropy \
                                            --label-smoothing 0.1 \
                                            --max-tokens 4096 \
                                            --eval-bleu  \
                                            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
                                            --eval-bleu-detok moses \
                                            --eval-bleu-remove-bpe \
                                            --eval-bleu-print-samples \
                                            --best-checkpoint-metric bleu \
                                            --maximize-best-checkpoint-metric \
                                            --patience 5  \
                                            --save-dir "experiment_outputs/${EXPERIMENT_NAME}" \
                                            --source-lang=$src \
                                            --target-lang=$tgt \
                                            --seed $SEED \
                                            --task "translation-with-subword-regularization" \
                                            --src-dropout $SRC_DROPOUT \
                                            --tgt-dropout $TGT_DROPOUT \
                                            --bpe-impl-path "/home/cognetta-m/github/fairseq_dropout_paper/fairseq/examples/translation_dropout/subword-nmt/subword_nmt" \
                                            --raw-data-path "/home/cognetta-m/github/fairseq_dropout_paper/fairseq/examples/translation_dropout/orig" \
                                            --no-epoch-checkpoints > experiment_outputs/${EXPERIMENT_NAME}/$EXPERIMENT_NAME.log
                                            


CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-generate examples/translation_dropout/data-bin/$EXPERIMENT_NAME \
                                        --path experiment_outputs/${EXPERIMENT_NAME}/checkpoint_best.pt \
                                        --batch-size 128 \
                                        --beam 5 \
                                        --max-len-a 1.2 \
                                        --max-len-b 10 \
                                        --remove-bpe > experiment_outputs/${EXPERIMENT_NAME}/bleu_unprocessed.log



cd experiment_outputs/${EXPERIMENT_NAME}

grep ^H bleu_unprocessed.log | cut -f3- > gen.out.sys
grep ^T bleu_unprocessed.log | cut -f2- > gen.out.ref
cat gen.out.sys | sacremoses -l en detokenize  > gen.out.sys.detok
cat gen.out.ref | sacremoses -l en detokenize  > gen.out.ref.detok
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m bleu -b -w 4 > BLEU.txt
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m chrf -b > CHRF.txt

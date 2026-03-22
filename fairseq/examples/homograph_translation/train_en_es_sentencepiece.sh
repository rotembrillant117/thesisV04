#!/usr/bin/env bash

#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

# Stop the bash script from running if anything fails
set -euo pipefail

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt

TOKENIZER_TYPE=bpe
SRC_BPE_TOKENS=8000
TGT_BPE_TOKENS=8000
SRC_DROPOUT=0.0
TGT_DROPOUT=0.0
SEED=0
DEVICE=0

EXPERIMENT_PREFIX="experiment"
# loops through the arguments and assign their values to bash parameters
while [[ "$#" -gt 0 ]]
do
    case $1 in
        --src-bpe-tokens) SRC_BPE_TOKENS=$2; shift ;;
        --tgt-bpe-tokens) TGT_BPE_TOKENS=$2; shift ;;
        --src-dropout) SRC_DROPOUT=$2; shift ;;
        --tgt-dropout) TGT_DROPOUT=$2; shift ;;
        --jamo-type) JAMO_TYPE=$2; shift ;;
        --tokenizer-type) TOKENIZER_TYPE=$2; shift ;;
        --seed) SEED=$2; shift ;;
        --device) DEVICE=$2; shift ;;
        --experiment-name) EXPERIMENT_PREFIX="$2"; shift ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ -z "${JAMO_TYPE:-}" ]]; then
    echo "Missing required argument: --jamo-type"
    exit 1
fi

echo "========= PARAMETERS =========== "
echo -e "JAMO_TYPE $JAMO_TYPE \nSRC_TOKENS $SRC_BPE_TOKENS \nTGT_TOKENS $TGT_BPE_TOKENS \nSRC_DROPOUT $SRC_DROPOUT \nTGT_DROPOUT $TGT_DROPOUT \nSEED $SEED \nDEVICE $DEVICE \nTOKENIZER_TYPE $TOKENIZER_TYPE \nNAME $EXPERIMENT_PREFIX\n"
echo "========= PARAMETERS =========== "

src=en
tgt=es
lang=en-es
# jamo type is either baseline or cues. this is an if statement that defines which datasets are used, depending on if it is a cued or baseline model
# and also which post_processing is used data_utils.py
case "$JAMO_TYPE" in
    en_es_baseline)
        MAIN_DATASET_DIR="orig/en_es_baseline"
        HOMOGRAPH_DATASET_DIR="orig/en_es_baseline_homograph"
        FLORES_DATASET_DIR="orig/en_es_flores"
        FLORES_HOMOGRAPH_DATASET_DIR="orig/en_es_flores_homograph"
        POST_PROCESS="sentencepiece"
        ;;
    en_es_cue)
        MAIN_DATASET_DIR="orig/en_es_cue"
        HOMOGRAPH_DATASET_DIR="orig/en_es_cue_homograph"
        FLORES_DATASET_DIR="orig/en_es_flores_cue"
        FLORES_HOMOGRAPH_DATASET_DIR="orig/en_es_flores_cue_homograph"
        POST_PROCESS="sentencepiece_cues"
        ;;
    *)
        echo "Unsupported --jamo-type: $JAMO_TYPE"
        echo "Expected one of: en_es_baseline, en_es_cue"
        exit 1
        ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_jamo_type_${JAMO_TYPE}_tokenizer_type_${TOKENIZER_TYPE}_${SRC_BPE_TOKENS}_${TGT_BPE_TOKENS}_dropout_${SRC_DROPOUT}_${TGT_DROPOUT}_seed_${SEED}.${lang}"

OUTPUT_ROOT="../../${src}_${tgt}_sentencepiece_experiment_outputs/${TOKENIZER_TYPE}"
EXPERIMENT_OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"

prep="experiments/$EXPERIMENT_NAME"
MAIN_DEST_REL="data-bin/$EXPERIMENT_NAME"

HOMOGRAPH_EVAL_NAME="${EXPERIMENT_NAME}_homograph_eval"
HOMOGRAPH_TEXT_REL="experiments/$HOMOGRAPH_EVAL_NAME"
HOMOGRAPH_DEST_REL="data-bin/$HOMOGRAPH_EVAL_NAME"

FLORES_EVAL_NAME="${EXPERIMENT_NAME}_flores_eval"
FLORES_TEXT_REL="experiments/$FLORES_EVAL_NAME"
FLORES_DEST_REL="data-bin/$FLORES_EVAL_NAME"

FLORES_HOMOGRAPH_EVAL_NAME="${EXPERIMENT_NAME}_flores_homograph_eval"
FLORES_HOMOGRAPH_TEXT_REL="experiments/$FLORES_HOMOGRAPH_EVAL_NAME"
FLORES_HOMOGRAPH_DEST_REL="data-bin/$FLORES_HOMOGRAPH_EVAL_NAME"

echo "MAIN_DATASET_DIR $MAIN_DATASET_DIR"
echo "HOMOGRAPH_DATASET_DIR $HOMOGRAPH_DATASET_DIR"
echo "FLORES_DATASET_DIR $FLORES_DATASET_DIR"
echo "FLORES_HOMOGRAPH_DATASET_DIR $FLORES_HOMOGRAPH_DATASET_DIR"
echo "POST_PROCESS $POST_PROCESS"
echo "EXPERIMENT_OUTPUT_DIR $EXPERIMENT_OUTPUT_DIR"

# if the experiment has already been done before, skip it
if [ -d "$EXPERIMENT_OUTPUT_DIR" ]
then
    echo "${EXPERIMENT_NAME} already done, SKIPPING"
    exit 0
fi
# before any tokenizer or model training or binarization happens, this big if block makes sure that all the files are present
if [[ ! -f "${MAIN_DATASET_DIR}/train.${src}" ]]; then
    echo "Missing main training source file: ${MAIN_DATASET_DIR}/train.${src}"
    exit 1
fi

if [[ ! -f "${MAIN_DATASET_DIR}/train.${tgt}" ]]; then
    echo "Missing main training target file: ${MAIN_DATASET_DIR}/train.${tgt}"
    exit 1
fi

if [[ ! -f "${MAIN_DATASET_DIR}/valid.${src}" ]]; then
    echo "Missing main validation source file: ${MAIN_DATASET_DIR}/valid.${src}"
    exit 1
fi

if [[ ! -f "${MAIN_DATASET_DIR}/valid.${tgt}" ]]; then
    echo "Missing main validation target file: ${MAIN_DATASET_DIR}/valid.${tgt}"
    exit 1
fi

if [[ ! -f "${MAIN_DATASET_DIR}/test.${src}" ]]; then
    echo "Missing main test source file: ${MAIN_DATASET_DIR}/test.${src}"
    exit 1
fi

if [[ ! -f "${MAIN_DATASET_DIR}/test.${tgt}" ]]; then
    echo "Missing main test target file: ${MAIN_DATASET_DIR}/test.${tgt}"
    exit 1
fi

if [[ ! -f "${HOMOGRAPH_DATASET_DIR}/test.${src}" ]]; then
    echo "Missing homograph source test file: ${HOMOGRAPH_DATASET_DIR}/test.${src}"
    exit 1
fi

if [[ ! -f "${HOMOGRAPH_DATASET_DIR}/test.${tgt}" ]]; then
    echo "Missing homograph target test file: ${HOMOGRAPH_DATASET_DIR}/test.${tgt}"
    exit 1
fi

if [[ ! -f "${FLORES_DATASET_DIR}/test.${src}" ]]; then
    echo "Missing FLORES source test file: ${FLORES_DATASET_DIR}/test.${src}"
    exit 1
fi

if [[ ! -f "${FLORES_DATASET_DIR}/test.${tgt}" ]]; then
    echo "Missing FLORES target test file: ${FLORES_DATASET_DIR}/test.${tgt}"
    exit 1
fi

if [[ ! -f "${FLORES_HOMOGRAPH_DATASET_DIR}/test.${src}" ]]; then
    echo "Missing FLORES homograph source test file: ${FLORES_HOMOGRAPH_DATASET_DIR}/test.${src}"
    exit 1
fi

if [[ ! -f "${FLORES_HOMOGRAPH_DATASET_DIR}/test.${tgt}" ]]; then
    echo "Missing FLORES homograph target test file: ${FLORES_HOMOGRAPH_DATASET_DIR}/test.${tgt}"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$prep"
mkdir -p "$MAIN_DEST_REL"
# trains a sentencepiece tokenizer with the relevant training data
python - <<PY
import sentencepiece as spm
spm.SentencePieceTrainer.Train(
    input="${MAIN_DATASET_DIR}/train.${src},${MAIN_DATASET_DIR}/train.${tgt}",
    model_prefix="$prep/joint_tokenizer",
    vocab_size=int("$SRC_BPE_TOKENS"),
    model_type="$TOKENIZER_TYPE",
)
PY
# saves the sentencepiece model files
python3 dump_unigram_vocab.py "$prep/joint_tokenizer.vocab" "$prep/joint_tokenizer.dict"
# for train, validation and test data, we load the trained tokenizer and tokenize all the "raw" data files, once for the
# source language (English), and once for the target language
for f in train valid test; do
    echo "encode (joint tokenizer) ($src) to ${f}.${src}..."
    python - <<PY
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="$prep/joint_tokenizer.model")
with open("${MAIN_DATASET_DIR}/$f.$src", "r", encoding="utf-8") as fin, open("$prep/$f.$src", "w", encoding="utf-8") as fout:
    for line in fin:
        pieces = sp.encode(line.strip(), out_type=str)
        fout.write(" ".join(pieces) + "\n")
PY
    cp "$prep/$f.$src" "$MAIN_DEST_REL/$f.$src"

    echo "encode (joint tokenizer) ($tgt) to ${f}.${tgt}..."
    python - <<PY
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="$prep/joint_tokenizer.model")
with open("${MAIN_DATASET_DIR}/$f.$tgt", "r", encoding="utf-8") as fin, open("$prep/$f.$tgt", "w", encoding="utf-8") as fout:
    for line in fin:
        pieces = sp.encode(line.strip(), out_type=str)
        fout.write(" ".join(pieces) + "\n")
PY
    cp "$prep/$f.$tgt" "$MAIN_DEST_REL/$f.$tgt"
done

cd ../..
# Convert the tokenized main train/valid/test text for this experiment into a fairseq
# data-bin dataset using the shared SentencePiece dictionary, then copy the tokenizer
# and dictionary files into the dataset directory so fairseq training/evaluation can use them.
TEXT="examples/homograph_translation/$prep"
MAIN_DEST="examples/homograph_translation/$MAIN_DEST_REL"
OUTPUT_DIR="${src}_${tgt}_sentencepiece_experiment_outputs/${TOKENIZER_TYPE}/${EXPERIMENT_NAME}"

fairseq-preprocess --source-lang "$src" --target-lang "$tgt" \
    --trainpref "$TEXT/train" --validpref "$TEXT/valid" --testpref "$TEXT/test" \
    --destdir "$MAIN_DEST" \
    --workers 8 \
    --joined-dictionary \
    --srcdict "$TEXT/joint_tokenizer.dict"

cp "$TEXT/joint_tokenizer.model" "$TEXT/joint_tokenizer.vocab" "$TEXT/joint_tokenizer.dict" "$MAIN_DEST/"
cp "$TEXT/joint_tokenizer.dict" "$MAIN_DEST/dict.${src}.txt"
cp "$TEXT/joint_tokenizer.dict" "$MAIN_DEST/dict.${tgt}.txt"

# Build the three evaluation-only fairseq datasets (OPUS homograph, FLORES full, FLORES homograph)
# by encoding each test set with the experiment's trained joint SentencePiece tokenizer and then
# preprocessing it with the same shared dictionary, so the trained model can be evaluated consistently
# on all test conditions without retraining or creating new tokenizers.

encode_test_pair() {
    local input_dir="$1"
    local output_dir="$2"

    rm -rf "$output_dir"
    mkdir -p "$output_dir"

    python - <<PY
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="$TEXT/joint_tokenizer.model")

for lang in ["$src", "$tgt"]:
    with open(f"$input_dir/test.{lang}", "r", encoding="utf-8") as fin, \
         open(f"$output_dir/test.{lang}", "w", encoding="utf-8") as fout:
        for line in fin:
            pieces = sp.encode(line.strip(), out_type=str)
            fout.write(" ".join(pieces) + "\n")
PY
}

prepare_test_only_eval_bin() {
    local eval_name="$1"
    local raw_test_dir="$2"
    local text_dir="examples/homograph_translation/experiments/$eval_name"
    local dest_dir="examples/homograph_translation/data-bin/$eval_name"

    rm -rf "$text_dir" "$dest_dir"
    mkdir -p "$text_dir" "$dest_dir"

    encode_test_pair "$raw_test_dir" "$text_dir"

    fairseq-preprocess --source-lang "$src" --target-lang "$tgt" \
        --testpref "$text_dir/test" \
        --destdir "$dest_dir" \
        --workers 8 \
        --joined-dictionary \
        --srcdict "$TEXT/joint_tokenizer.dict"

    cp "$TEXT/joint_tokenizer.model" "$TEXT/joint_tokenizer.vocab" "$TEXT/joint_tokenizer.dict" "$dest_dir/"
    cp "$TEXT/joint_tokenizer.dict" "$dest_dir/dict.${src}.txt"
    cp "$TEXT/joint_tokenizer.dict" "$dest_dir/dict.${tgt}.txt"
}

prepare_test_only_eval_bin "$HOMOGRAPH_EVAL_NAME" "examples/homograph_translation/${HOMOGRAPH_DATASET_DIR}"
prepare_test_only_eval_bin "$FLORES_EVAL_NAME" "examples/homograph_translation/${FLORES_DATASET_DIR}"
prepare_test_only_eval_bin "$FLORES_HOMOGRAPH_EVAL_NAME" "examples/homograph_translation/${FLORES_HOMOGRAPH_DATASET_DIR}"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$DEVICE fairseq-train "examples/homograph_translation/data-bin/$EXPERIMENT_NAME" \
    --arch transformer_iwslt_de_en \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --validate-after-updates 1000 \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-detok-args '{"target_lang": "es"}' \
    --eval-bleu-remove-bpe="${POST_PROCESS}" \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --patience 8 \
    --save-dir "$OUTPUT_DIR" \
    --source-lang="$src" \
    --target-lang="$tgt" \
    --seed "$SEED" \
    --task "translation-with-subword-regularization" \
    --src-dropout "$SRC_DROPOUT" \
    --tgt-dropout "$TGT_DROPOUT" \
    --jamo-type "$JAMO_TYPE" \
    --bpe-impl-path "$(pwd)/examples/homograph_translation" \
    --raw-data-path "$(pwd)/examples/homograph_translation/${MAIN_DATASET_DIR}" \
    --no-epoch-checkpoints \
    > "${OUTPUT_DIR}/${EXPERIMENT_NAME}.log"

CUDA_VISIBLE_DEVICES=$DEVICE fairseq-generate "examples/homograph_translation/data-bin/$EXPERIMENT_NAME" \
    --path "${OUTPUT_DIR}/checkpoint_best.pt" \
    --batch-size 128 \
    --beam 5 \
    --max-len-a 1.2 \
    --max-len-b 10 \
    --remove-bpe="${POST_PROCESS}" \
    --source-lang="$src" \
    --target-lang="$tgt" \
    > "${OUTPUT_DIR}/bleu_unprocessed.log"

CUDA_VISIBLE_DEVICES=$DEVICE fairseq-generate "examples/homograph_translation/data-bin/$HOMOGRAPH_EVAL_NAME" \
    --path "${OUTPUT_DIR}/checkpoint_best.pt" \
    --batch-size 128 \
    --beam 5 \
    --max-len-a 1.2 \
    --max-len-b 10 \
    --remove-bpe="${POST_PROCESS}" \
    --source-lang="$src" \
    --target-lang="$tgt" \
    > "${OUTPUT_DIR}/bleu_homograph_unprocessed.log"

CUDA_VISIBLE_DEVICES=$DEVICE fairseq-generate "examples/homograph_translation/data-bin/$FLORES_EVAL_NAME" \
    --path "${OUTPUT_DIR}/checkpoint_best.pt" \
    --batch-size 128 \
    --beam 5 \
    --max-len-a 1.2 \
    --max-len-b 10 \
    --remove-bpe="${POST_PROCESS}" \
    --source-lang="$src" \
    --target-lang="$tgt" \
    > "${OUTPUT_DIR}/bleu_flores_unprocessed.log"

CUDA_VISIBLE_DEVICES=$DEVICE fairseq-generate "examples/homograph_translation/data-bin/$FLORES_HOMOGRAPH_EVAL_NAME" \
    --path "${OUTPUT_DIR}/checkpoint_best.pt" \
    --batch-size 128 \
    --beam 5 \
    --max-len-a 1.2 \
    --max-len-b 10 \
    --remove-bpe="${POST_PROCESS}" \
    --source-lang="$src" \
    --target-lang="$tgt" \
    > "${OUTPUT_DIR}/bleu_flores_homograph_unprocessed.log"

cd "$OUTPUT_DIR"

grep --text ^H bleu_unprocessed.log | cut -f3- > gen.out.sys
grep --text ^T bleu_unprocessed.log | cut -f2- > gen.out.ref
cat gen.out.sys | sacremoses -l es detokenize > gen.out.sys.detok
cat gen.out.ref | sacremoses -l es detokenize > gen.out.ref.detok
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m bleu -b -w 4 > BLEU.txt
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m chrf -b > CHRF.txt

grep --text ^H bleu_homograph_unprocessed.log | cut -f3- > gen.out.homograph.sys
grep --text ^T bleu_homograph_unprocessed.log | cut -f2- > gen.out.homograph.ref
cat gen.out.homograph.sys | sacremoses -l es detokenize > gen.out.homograph.sys.detok
cat gen.out.homograph.ref | sacremoses -l es detokenize > gen.out.homograph.ref.detok
sacrebleu gen.out.homograph.ref.detok -i gen.out.homograph.sys.detok -m bleu -b -w 4 > BLEU_homograph.txt
sacrebleu gen.out.homograph.ref.detok -i gen.out.homograph.sys.detok -m chrf -b > CHRF_homograph.txt

grep --text ^H bleu_flores_unprocessed.log | cut -f3- > gen.out.flores.sys
grep --text ^T bleu_flores_unprocessed.log | cut -f2- > gen.out.flores.ref
cat gen.out.flores.sys | sacremoses -l es detokenize > gen.out.flores.sys.detok
cat gen.out.flores.ref | sacremoses -l es detokenize > gen.out.flores.ref.detok
sacrebleu gen.out.flores.ref.detok -i gen.out.flores.sys.detok -m bleu -b -w 4 > BLEU_flores.txt
sacrebleu gen.out.flores.ref.detok -i gen.out.flores.sys.detok -m chrf -b > CHRF_flores.txt

grep --text ^H bleu_flores_homograph_unprocessed.log | cut -f3- > gen.out.flores.homograph.sys
grep --text ^T bleu_flores_homograph_unprocessed.log | cut -f2- > gen.out.flores.homograph.ref
cat gen.out.flores.homograph.sys | sacremoses -l es detokenize > gen.out.flores.homograph.sys.detok
cat gen.out.flores.homograph.ref | sacremoses -l es detokenize > gen.out.flores.homograph.ref.detok
sacrebleu gen.out.flores.homograph.ref.detok -i gen.out.flores.homograph.sys.detok -m bleu -b -w 4 > BLEU_flores_homograph.txt
sacrebleu gen.out.flores.homograph.ref.detok -i gen.out.flores.homograph.sys.detok -m chrf -b > CHRF_flores_homograph.txt
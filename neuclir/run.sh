#! /bin/env bash

set -ex

INPUT_DIR='/net/nfs.cirrascale/s2-research/lucas/neuclir/2023/cls/encodings'
OUTPUT_DIR='/net/nfs.cirrascale/s2-research/lucas/neuclir/2023/cls/results'

mkdir -p "${OUTPUT_DIR}"

# 1 Monolingual run
# A Automatic run
# T Translated documents
# D dense retrieval
# E English queries

RUN_NAME="tech-AI2-1DEAT-Specter2Title"

python -m tevatron.faiss_retriever \
    --query_reps "${INPUT_DIR}/topic_title.pkl" \
    --passage_reps "${INPUT_DIR}/documents.pkl" \
    --depth 1000 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to "${OUTPUT_DIR}/${RUN_NAME}.txt"

python -m tevatron.utils.format.convert_result_to_trec \
    --input "${OUTPUT_DIR}/${RUN_NAME}.txt" \
    --output "${OUTPUT_DIR}/${RUN_NAME}.trec" \
    --name "${RUN_NAME}"

#######################

RUN_NAME="tech-AI2-1DEAT-Specter2Description"

python -m tevatron.faiss_retriever \
    --query_reps "${INPUT_DIR}/topic_description.pkl" \
    --passage_reps "${INPUT_DIR}/documents.pkl" \
    --depth 1000 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to "${OUTPUT_DIR}/${RUN_NAME}.txt"

python -m tevatron.utils.format.convert_result_to_trec \
    --input "${OUTPUT_DIR}/${RUN_NAME}.txt" \
    --output "${OUTPUT_DIR}/${RUN_NAME}.trec" \
    --name "${RUN_NAME}"

#######################

RUN_NAME="tech-AI2-1DEAT-Specter2Description"

python -m tevatron.faiss_retriever \
    --query_reps "${INPUT_DIR}/topic_narrative.pkl" \
    --passage_reps "${INPUT_DIR}/documents.pkl" \
    --depth 1000 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to "${OUTPUT_DIR}/${RUN_NAME}.txt"

python -m tevatron.utils.format.convert_result_to_trec \
    --input "${OUTPUT_DIR}/${RUN_NAME}.txt" \
    --output "${OUTPUT_DIR}/${RUN_NAME}.trec" \
    --name "${RUN_NAME}"

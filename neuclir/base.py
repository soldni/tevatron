import logging
from typing import Tuple

import torch
from transformers import AutoAdapterModel, HfArgumentParser

from tevatron.arguments import DataArguments, ModelArguments
from tevatron.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.datasets.dataset import PROCESSOR_INFO
from tevatron.datasets.preprocessor import (
    CorpusPreProcessor,
    QueryPreProcessor,
    TrainPreProcessor,
)
from tevatron.modeling import DenseModel


class SpecterModel(DenseModel):
    TRANSFORMER_CLS = AutoAdapterModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lm_q.load_adapter(
            "allenai/specter2_adhoc_query", source="hf", set_active=True
        )  # type: ignore

        self.lm_p.load_adapter(
            "allenai/specter2_proximity", source="hf", set_active=True
        )  # type: ignore

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        return psg_out.last_hidden_state[:, 0, :]

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        return qry_out.last_hidden_state[:, 0, :]

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))


class NeuclirCorpusPreprocessor(CorpusPreProcessor):
    def __call__(self, example: dict) -> dict:
        docid = example["doc_id"]
        text = " ".join(
            (example["title"], self.tokenizer.sep_token, example.get("abstract", ""))
        ).strip()
        text = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.text_max_length,
            truncation=True,
        )
        return {"text_id": docid, "text": text}


class NeuclirQueryPreprocessor(QueryPreProcessor):
    def __call__(self, example: dict) -> dict:
        import ipdb

        ipdb.set_trace()
        query_id = example["query_id"]
        query = self.tokenizer.encode(
            example["query"],
            add_special_tokens=False,
            max_length=self.query_max_length,
            truncation=True,
        )
        return {"text_id": query_id, "text": query}


PROCESSOR_INFO["neuclir/csl"] = [
    TrainPreProcessor,
    NeuclirQueryPreprocessor,
    NeuclirCorpusPreprocessor,
]


def parse_arguments(
    **kwargs,
) -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)  # type: ignore
    )
    model_args, data_args, training_args = parser.parse_dict(kwargs)
    return model_args, data_args, training_args


def setup_logger(rank: int):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
    )

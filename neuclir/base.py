import logging
from dataclasses import dataclass, field
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

        self.use_passage_enc_for_query = False

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        return psg_out.last_hidden_state[:, 0, :]

    def encode_query(self, qry):
        if qry is None:
            return None

        encoder = self.lm_p if self.use_passage_enc_for_query else self.lm_q
        qry_out = encoder(**qry, return_dict=True)
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
    def __init__(
        self, tokenizer, query_field: str = "topic_title", query_max_length: int = 32
    ):
        super().__init__(tokenizer=tokenizer, query_max_length=query_max_length)

    @property
    def query_field(self):
        return getattr(self, "_query_field", "topic_title")

    @query_field.setter
    def query_field(self, value):
        self._query_field = value

    def __call__(self, example: dict) -> dict:
        query_id = example["topic_id"]
        eng_queries = [q for q in example["topics"] if q["lang"] == "eng"]
        assert len(eng_queries) == 1, "Only one English query is supported"

        query = self.tokenizer.encode(
            eng_queries[0][self.query_field],
            add_special_tokens=False,
            max_length=self.query_max_length,
            truncation=True,
        )
        return {"text_id": query_id, "text": query}


PROCESSORS = [
    TrainPreProcessor,
    NeuclirQueryPreprocessor,
    NeuclirCorpusPreprocessor,
]
PROCESSOR_INFO["neuclir/csl"] = PROCESSORS
PROCESSOR_INFO["neuclir/csl-topics"] = PROCESSORS


@dataclass
class NeuclirDataArguments(DataArguments):
    query_field: str = field(
        default="topic_title", metadata={"help": "Field to use as query"}
    )


@dataclass
class NeuclirModelArguments(ModelArguments):
    use_passage_enc_for_query: bool = field(
        default=False, metadata={"help": "Use passage encoder for query"}
    )


def parse_arguments(
    **kwargs,
) -> Tuple[NeuclirModelArguments, NeuclirDataArguments, TrainingArguments]:
    parser = HfArgumentParser(
        (NeuclirModelArguments, NeuclirDataArguments, TrainingArguments)  # type: ignore
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

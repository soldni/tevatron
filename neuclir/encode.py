import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from torch import nn
from transformers import AutoAdapterModel
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel
from transformers import (
    HfArgumentParser,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import EncodeDataset, EncodeCollator
from tevatron.modeling import EncoderOutput, DenseModel
from tevatron.datasets import HFQueryDataset, HFCorpusDataset

logger = logging.getLogger(__name__)


class SpecterModel(DenseModel):
    TRANSFORMER_CLS = AutoAdapterModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lm_q.load_adapter(
            "allenai/specter2_adhoc_query", source="hf", set_active=True
        )   # type: ignore

        self.lm_p.load_adapter(
            "allenai/specter2_proximity", source="hf", set_active=True
        )   # type: ignore

    def encode_passage(self, psg):

        import ipdb; ipdb.set_trace()

        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden[:, 0]
        return q_reps

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))


BASE_DIR = "/net/nfs.cirrascale/s2-research/lucas/neuclir/2023/cls"


def main(
    base_dir: str = BASE_DIR,
    **kwargs
):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict({
        "model_name_or_path": "allenai/specter2_base",
        "dataset_name": "neuclir/csl/en_translation",
        "output_dir": os.path.join(base_dir, "encoded"),
        "add_pooler": False,
    })
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    model = SpecterModel.load(
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    text_max_length = (
        data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    )
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir
        )
    else:
        encode_dataset = HFCorpusDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir
        )
    encode_dataset = EncodeDataset(
        dataset=encode_dataset.process(
            shard_num=data_args.encode_num_shard,
            shard_idx=data_args.encode_shard_index
        ),
        tokenizer=tokenizer,
        max_len=text_max_length
    )

    encode_loader = DataLoader(
        dataset=encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    model_output: EncoderOutput = model(query=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    model_output: EncoderOutput = model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())

    encoded = np.concatenate(encoded)

    with open(data_args.encoded_save_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()

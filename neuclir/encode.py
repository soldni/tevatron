import os
import pickle
from contextlib import nullcontext
from functools import partialmethod
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from tevatron.data import EncodeCollator, EncodeDataset
from tevatron.datasets import HFCorpusDataset, HFQueryDataset
from tevatron.modeling import EncoderOutput

from .base import SpecterModel, parse_arguments, setup_logger


def main(
    base_dir: str = "/net/nfs.cirrascale/s2-research/lucas/neuclir/2023/cls",
    encoded_name: str = "encoded.pkl",
    **kwargs
):
    kwargs = {
        "model_name_or_path": "allenai/specter2_base",
        "dataset_name": "neuclir/csl/en_translation",
        "output_dir": base_dir,
        "encoded_save_path": os.path.join(base_dir, encoded_name),
        "add_pooler": False,
        # "dataset_proc_num": None,
        "p_max_len": 512,
        "per_device_eval_batch_size": 512,
        "fp16": True,
        **kwargs,
    }
    model_args, data_args, training_args = parse_arguments(**kwargs)
    setup_logger(rank=training_args.local_rank)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
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
            cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        )
        encode_dataset.query_field = data_args.query_field
    else:
        encode_dataset = HFCorpusDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        )
    encode_dataset = EncodeDataset(
        dataset=encode_dataset.process(
            shard_num=data_args.encode_num_shard, shard_idx=data_args.encode_shard_index
        ),
        tokenizer=tokenizer,
        max_len=text_max_length,
    )

    encode_loader = DataLoader(
        dataset=encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(
            tokenizer, max_length=text_max_length, padding="max_length"
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    if training_args.fp16:
        model = model.to(torch.bfloat16)
    model.eval()

    Path(data_args.encoded_save_path).parent.mkdir(parents=True, exist_ok=True)

    for batch_ids, batch in tqdm(encode_loader):
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

    with open(data_args.encoded_save_path, "wb") as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()

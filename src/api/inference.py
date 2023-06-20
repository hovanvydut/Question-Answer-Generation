from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    T5Tokenizer
)

import argparse
import pandas as pd
import sys
import os
sys.path.append("../")
from utils import currentdate, QAG_TAG, QG_TAG, AE_TAG

from model.model_qag import QAGModel
from utils import currentdate
import time
import json
import torch

def generate(args, device, qgmodel: QAGModel, tokenizer: T5Tokenizer,  context: str) -> str:

    source_encoding = tokenizer(
        context,
        max_length=args.max_len_input,
        padding='max_length',
        truncation = 'only_second',
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    input_ids = source_encoding['input_ids'].to(device)
    attention_mask = source_encoding['attention_mask'].to(device)

    generated_ids = qgmodel.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_return_sequences=args.num_return_sequences,
        num_beams=args.num_beams, 
        max_length=args.max_len_output,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    
    return '<DIV>'.join(preds)

params_dict = dict(
    checkpoint_model_path = "../../checkpoints/qag/model-epoch=01-val_loss=1.14.ckpt",
    model_name = "google/mt5-base",
    tokenizer_name = "google/mt5-base",
    max_len_input = 512,
    max_len_output = 96,
    num_beams = 5,
    num_return_sequences = 5,
    repetition_penalty = 1.0,
    length_penalty = 1.0,
)
params = argparse.Namespace(**params_dict)

t5_tokenizer = T5Tokenizer.from_pretrained(params.tokenizer_name)
if "mt5" in params.model_name:
    t5_model = MT5ForConditionalGeneration.from_pretrained(params.model_name)
else:
    t5_model = T5ForConditionalGeneration.from_pretrained(params.model_name)

checkpoint_model_path = params.checkpoint_model_path
qgmodel = QAGModel.load_from_checkpoint(checkpoint_model_path, hparams=params, t5model=t5_model, t5tokenizer=t5_tokenizer)

qgmodel.freeze()
qgmodel.eval()

# Put model in gpu (if possible) or cpu (if not possible) for inference purpose
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qgmodel = qgmodel.to(device)
print ("Device for inference:", device)
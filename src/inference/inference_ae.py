from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    T5Tokenizer
)

import argparse
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_ae import AEModel
from utils import currentdate
import time
import json
import torch

def generate(args, device, qgmodel: AEModel, tokenizer: T5Tokenizer,  context: str) -> str:

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
    
    return '|'.join(preds)


def run(args):
    params_dict = dict(
        checkpoint_model_path = args.checkpoint_model_path,
        model_name = args.model_name,
        tokenizer_name = args.tokenizer_name,
        max_len_input = args.max_len_input,
        max_len_output = args.max_len_output,
        num_beams = args.num_beams,
        num_return_sequences = args.num_return_sequences,
        repetition_penalty = args.repetition_penalty,
        length_penalty = args.length_penalty,
    )
    params = argparse.Namespace(**params_dict)

    t5_tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
    if "mt5" in args.model_name:
        t5_model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        t5_model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    checkpoint_model_path = args.checkpoint_model_path
    qgmodel = AEModel.load_from_checkpoint(checkpoint_model_path, hparams=params, t5model=t5_model, t5tokenizer=t5_tokenizer)

    qgmodel.freeze()
    qgmodel.eval()

    # Put model in gpu (if possible) or cpu (if not possible) for inference purpose
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qgmodel = qgmodel.to(device)
    print ("Device for inference:", device)

    # for loop
    context = ""
    code = ""
    while code != "q":
        print("===============")
        context = input("nhap context:")

        st = time.time()
        generated = generate(args, device, qgmodel, t5_tokenizer, context)
        et = time.time()
        
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')

        print("Ket qua: answer = ")
        print("---------------")
        print(generated)
        print("---------------")
        code = input("Nhan phim q de thoat, nhap phim bat ki de tiep tuc")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate questions and save them to json file.')

    parser.add_argument('-cmp','--checkpoint_model_path', type=str, metavar='', default="../../checkpoints/ae/model-epoch=03-val_loss=1.32.ckpt", required=False, help='Model folder checkpoint path.')

    parser.add_argument('-mn','--model_name', type=str, metavar='', default="google/mt5-base", required=False, help='Model name.')
    parser.add_argument('-tn','--tokenizer_name', type=str, metavar='', default="google/mt5-base", required=False, help='Tokenizer name.')

    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=96, required=False, help='Max len output for encoding.')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=5, required=False, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=5, required=False, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')

    args = parser.parse_args()

    run(args)
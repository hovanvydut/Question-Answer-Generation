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

df_test = pd.read_pickle("../../data/squad_vi/raw/dataframe/df_test_en.pkl")

# Customize tokenizer behavior for Vietnamese
t5_tokenizer.do_lower_case = True  # Set to True if you want lowercase tokens
t5_tokenizer.remove_space = True  # Set to True if you want to remove leading/trailing spaces

import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.metrics.scores import (precision, recall)

custom_config = params_dict.copy()
custom_config["num_return_sequences"] = 1
custom_config = argparse.Namespace(**custom_config)

import pickle
start_idx = 0
n = df_test.shape[0]
# n = 42
data_test = []
c = 0
generated_questions = []

# Load the existing data from the pickle file, if it exists
try:
    with open('./generated_data.pkl', 'rb') as f:
        generated_data = pickle.load(f)
        generated_questions = [row[3] for row in generated_data]
        start_idx = len(generated_data)
except FileNotFoundError:
    generated_data = []

for i in range(start_idx, n):
    context_ref = df_test.loc[i]["context"]
    answer_ref = df_test.loc[i]["answer"]
    question_ref = df_test.loc[i]["question"]

    source_text = f"{QG_TAG} answer: {answer_ref} context: {context_ref}"
    question_predicted = generate(custom_config, device, qgmodel, t5_tokenizer, source_text)
    data_test.append([context_ref, answer_ref, question_ref, question_predicted])
    generated_questions.append(question_predicted)
    c += 1
    if c >= 10:
        # Append the newly generated data to the existing data
        generated_data.extend(data_test)
        data_test = []

        # Save the updated data to the pickle file
        with open('generated_data.pkl', 'wb') as f:
            pickle.dump(generated_data, f)
        c = 0
    print(f"Generated question i-th = {i}")
# Append the remaining data to the existing data
generated_data.extend(data_test)

# Save the final set of generated data to the pickle file
with open('generated_data.pkl', 'wb') as f:
    pickle.dump(generated_data, f)
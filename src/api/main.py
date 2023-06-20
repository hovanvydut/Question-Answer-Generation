from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
sys.path.append("../")
from inference import qgmodel, t5_tokenizer, generate, device, params
from utils import QAG_TAG, QG_TAG, AE_TAG
import time

app = FastAPI()

class ReqData(BaseModel):
    task: str
    context: str
    question: str
    answer: str
    num_return: int

@app.post("/predict")
async def question_answer_generation(data: ReqData):
    source_text = ""
    if data.task == QG_TAG:
        source_text = f"{QG_TAG} answer: {data.answer} context: {data.context}"
    elif data.task == AE_TAG:
        source_text = f"{AE_TAG} context: {data.context}"
    elif data.task == QAG_TAG:
        source_text = f"{QAG_TAG} context: {data.context}"
    else:
        return {"error": "task id is invalid, must be QAG, EA, QG"}
    try:
        custom_config = params.copy()
        if data.num_return is not None:
            custom_config.num_return_sequences = data.num_return
        
        st = time.time()
        generated = generate(custom_config, device, qgmodel, t5_tokenizer, source_text)
        et = time.time()
        elapsed_time = et - st
        
        return {
            "task": data.task,
            "result": generated,
            "exec_time": elapsed_time
        }
    except Exception as e:
        return {"error": e}
    
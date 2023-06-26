from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import sys
import os
import argparse
sys.path.append("../")
from inference import qgmodel, t5_tokenizer, generate, device, params, params_dict
from utils import QAG_TAG, QG_TAG, AE_TAG
import time

app = FastAPI()

class ReqData(BaseModel):
    task: Optional[str]
    context: Optional[str]
    question: Optional[str]
    answer: Optional[str]
    num_return: Optional[int]

class ReqAEData(BaseModel):
    context: str
    num_return: int

class ReqQGData(BaseModel):
    context: str
    answer: str
    num_return: int

class ReqQAGData(BaseModel):
    context: str
    num_return: int

class ReqPipeData(BaseModel):
    context: str

@app.get("/ping")
async def ping():
    return "Ok"

@app.post("/predict/ae")
async def answer_extraction(data: ReqAEData):
    dto = ReqData()
    dto.task = AE_TAG
    dto.context = data.context
    dto.num_return = data.num_return
    return question_answer_generation(dto)

@app.post("/predict/qg")
async def answer_extraction(data: ReqQGData):
    dto = ReqData()
    dto.task = QG_TAG
    dto.context = data.context
    dto.answer = data.answer
    dto.num_return = data.num_return
    return question_answer_generation(dto)

@app.post("/predict/qag")
async def answer_extraction(data: ReqQAGData):
    dto = ReqData()
    dto.task = QAG_TAG
    dto.context = data.context
    dto.num_return = data.num_return
    return question_answer_generation(dto)

@app.post("/predict/pipeline")
async def answer_extraction(data: ReqPipeData):
    dto1 = ReqData()
    dto1.task = AE_TAG
    dto1.context = data.context
    dto1.num_return = 5
    answer_text:str = question_answer_generation(dto1)["result"]

    answers = answer_text.split("<DIV>")
    answers = [x for x in answers if x != '']
    answers = list(set(answers))

    result = []
    for answer in answers:
        dto2 = ReqData()
        dto2.task = QG_TAG
        dto2.context = data.context
        dto2.answer = answer
        dto2.num_return = 1
        question = question_answer_generation(dto2)
        result.append({"question": question, "answer": answer})
    
    return result

def question_answer_generation(data: ReqData):
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
        custom_config = params_dict.copy()
        if data.num_return is None:
            data.num_return = 1
        custom_config["num_return_sequences"] = data.num_return

        custom_config = argparse.Namespace(**custom_config)
        print(custom_config)
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
        print(e)
        return {"error": str(e)}
    
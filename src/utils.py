from datetime import datetime

def currentdate():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

QG_TAG = "question_generation"
QAG_TAG = "question_answer_generation"
AE_TAG = "answer_extraction"
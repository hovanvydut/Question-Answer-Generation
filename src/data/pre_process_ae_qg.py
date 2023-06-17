import pandas as pd

import json
import sys
sys.path.append('../')

# Preprocess data for Answer Extraction Task and Question Generation Task
with open('../../data/squad_vi/raw/json/train.json') as train_json_file:
    train_data = json.load(train_json_file)

with open('../../data/squad_vi/raw/json/dev.json') as dev_json_file:
    validation_data = json.load(dev_json_file)


train_all_compiled = []
for document in train_data:
    train_all_compiled.append([document[0], document[1], document[2]])
train_df = pd.DataFrame(train_all_compiled, columns = ['context', 'question', 'answer'])
print("Train Dataframe completed.")

TEST_TRAIN_RATE = 0.2
data_num_row = train_df.shape[0]
mid_idx = int(data_num_row * TEST_TRAIN_RATE)
test_df = train_df[:mid_idx]
train_df = train_df[mid_idx:]
print("Test Dataframe completed.")

val_all_compiled = []
for document in validation_data:
    val_all_compiled.append([document[0], document[1], document[2]])
validation_df = pd.DataFrame(val_all_compiled, columns = ['context', 'question', 'answer'])
print("Validation Dataframe completed.")


print("\n")
print("Number of train para-question-answer pairs: ", len(train_df))
print("Number of validation para-question-answer: ", len(validation_df))
print("Number of test para-question-answer: ", len(test_df))

train_df.to_pickle("../../data/squad_vi/raw/dataframe/df_train_vi.pkl")
validation_df.to_pickle("../../data/squad_vi/raw/dataframe/df_validation_vi.pkl")
test_df.to_pickle("../../data/squad_vi/raw/dataframe/df_test_vi.pkl")
print("Pickles were generated from dataframes.")

import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
from scipy.spatial.distance import cosine
from Preprocessing.preprocess_text import *

"""
model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3)
output_model_dir = "./new_one/"
model.save_pretrained(output_model_dir)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
tokenizer.save_pretrained(output_model_dir)
"""
file2 = "C:/Users/nikch/Text_analysis/Texts/2_harry.docx"
file1 = "C:/Users/nikch/Text_analysis/Texts/1_harry.docx"
dicted_file1 = read_docx(file_path = file1, language = 'russian')
dicted_file2 = read_docx(file_path = file2, language = 'russian')
#preprocessed_text1 = preprocess_text(text = dicted_file1['text'], language = "russian")
#preprocessed_text2 = preprocess_text(text = dicted_file2['text'], language = "russian")

#text1 = "Конечно, руководитель этого органа находится в постоянном прямом контакте с главой государства и несет большую ответственность"
#text2 = "Руководитель этого органа не находится в постоянном прямом контакте с главой государства и не несет никакой ответственности"

preprocessed_text1 = preprocess_text(text = dicted_file1["text"], language = "russian")
preprocessed_text2 = preprocess_text(text = dicted_file2["text"], language = "russian")
#print(preprocessed_text1)

preprocessed_cut1 = preprocessed_text1[:210]
preprocessed_cut2 = ""

output_model_dir = "./new_one/"
model = RobertaForSequenceClassification.from_pretrained(output_model_dir)
tokenizer = RobertaTokenizer.from_pretrained(output_model_dir)

def compare_texts(text1, text2):

    inputs = tokenizer(text1, text2, return_tensors="pt")


    outputs = model(**inputs)


    predicted_class = torch.argmax(outputs.logits).item()
    
    return predicted_class


predicted_similarity = compare_texts(preprocessed_cut1, preprocessed_cut2)

print(f"Сходство текстов: {predicted_similarity}")
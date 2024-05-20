from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
import pandas as pd
from datasets import load_dataset

from Preprocessing.preprocess_text import *

#file2 = "C:/Users/nikch/Text_analysis/Texts/Blow_internet.docx"
#file1 = "C:/Users/nikch/Text_analysis/Texts/10_End_of_the_Game.docx"
#dicted_file1 = read_docx(file_path = file1, language = 'english')
#dicted_file2 = read_docx(file_path = file2, language = 'english')
#preprocessed_text1 = preprocess_text(text = dicted_file1['text'], language = "english")
#preprocessed_text2 = preprocess_text(text = dicted_file2['text'], language = "english")

file2 = "C:/Users/nikch/Text_analysis/Texts/2_harry.docx"
file1 = "C:/Users/nikch/Text_analysis/Texts/1_harry.docx"

dicted_file1 = read_docx(file_path = file1, language = 'russian')
dicted_file2 = read_docx(file_path = file2, language = 'russian')
preprocessed_text1 = preprocess_text(text = dicted_file1['text'], language = "russian")
preprocessed_text2 = preprocess_text(text = dicted_file2['text'], language = "russian")

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

# Получение векторных представлений для обоих текстов
embeddings = model.encode([dicted_file1["text"], dicted_file1["text"]], convert_to_tensor=True)


# Вычисление косинусного расстояния между векторами
similarity = 1 - cosine(embeddings[0], embeddings[1])

print(similarity)



from transformers import pipeline
from Preprocessing.preprocess_text import *
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math


def softmax(x,bottom):
    return np.exp(x) / bottom

def softmax_bottom(x):
    return np.sum(np.exp(x), axis=0)


def tokenize_sentences(text):

    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    sentences = sentence_endings.split(text)
    
    return sentences

model = pipeline(model="seara/rubert-tiny2-ru-go-emotions")
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")


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

sentences1 =  tokenize_sentences(dicted_file1["text"])
sentences2 =  tokenize_sentences(dicted_file2["text"])

dict_res1 = defaultdict()
dict_avg1 = defaultdict()
dict_percent1 = {}
dict_res2 = defaultdict()
dict_avg2 = defaultdict()
for i in sentences1:
    #print(i)
    #print(model(i))
    dict_res1[model(i)[0]["label"]] = dict_res1.get(model(i)[0]["label"],0) + 1
    dict_avg1[model(i)[0]["label"]] = dict_avg1.get(model(i)[0]["label"],0) + model(i)[0]["score"]
    #print(model(i)[0]["score"])

for i in sentences2:
    #print(i)
    #print(model(i))
    dict_res2[model(i)[0]["label"]] = dict_res2.get(model(i)[0]["label"],0) + 1
    dict_avg2[model(i)[0]["label"]] = dict_avg2.get(model(i)[0]["label"],0) + model(i)[0]["score"]
print(dict_avg1,"\n")

bottom1 = sum(list(dict_res1.values()))
bottom2 = sum(list(dict_res2.values()))

for i in dict_avg1.keys():
    dict_avg1[i] = dict_avg1[i] / dict_res1[i]

for i in dict_avg2.keys():
    dict_avg2[i] = dict_avg2[i] / dict_res2[i]
    
for i in dict_res1.keys():
    dict_percent1[i] = dict_percent1.get(i,0) + (math.floor((dict_res1[i] / bottom1) * 1000) // 1000) * 100 



print("Базовый словарь")
print(dict_res1,"\n")
print("Среднее значение каждой эмоции")
print(dict_avg1,"\n")
print("Процент содержания каждой эмоции")
print(dict_percent1,"\n")
print(sum(list(dict_percent1.values())))

bottom1 = sum(list(dict_res1.values()))
bottom2 = sum(list(dict_res2.values()))

#print(sum(list(dict_res1.values())))
#print(sum(list(dict_res2.values())))
#print(dict_res2)
#plt.plot(list(dict_res1.values()))
res_base = {}
for i in dict_res1:
    if dict_res2[i]:
        pass
    else:
        res_base[i] = pow(dict_res1[i],2)

for i in dict_res2:
    res_base[i] = pow((dict_res1.get(i,0)-dict_res2[i]),2)
    
print("Дельта")
print(res_base,"\n")
print(sum(list(res_base.values())),"\n\n\n")


bottom1 = softmax_bottom(list(dict_res1.values()))
bottom2 = softmax_bottom(list(dict_res2.values()))


for i in dict_res1:
    dict_res1[i] = np.exp(dict_res1[i]) / bottom1
    
for i in dict_res2:
    dict_res2[i] = np.exp(dict_res2[i]) / bottom2   


#print(dict_res1)
#print(dict_res2)

res_dict = {}
for i in dict_res1:
    if dict_res2[i]:
        pass
    else:
        res_dict[i] = pow(dict_res1[i],2)

for i in dict_res2:
    res_dict[i] = pow((dict_res1.get(i,0)-dict_res2[i]),2)   

    
print("Нормализованная дельта")
print(res_dict,"\n")
print(sum(list(res_dict.values())))
#plt.plot(list(dict_res1.values()))
#print(classifier(preprocessed_text2))
#print(model("Кроме того, Путин попросил членов правительства учесть предложения депутатов Госдумы, которые прозвучали в ходе обсуждений, а также в короткие сроки полностью сформировать команду в министерствах и ведомствах. Он напомнил, что работу нужно выстраивать на шестилетку, а в ближайшее время необходимо определить финансовое обеспечение планов по поддержке семей."))
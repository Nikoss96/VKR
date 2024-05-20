import pandas as pd
from nltk.translate import bleu_score
from nltk.translate import meteor_score
from docx import Document
import docx
from Models.Bert import *
from datetime import date
from Preprocessing.preprocess_text import *
from datetime import datetime
import time
from Metrics.bm25 import *

def save_dataframe_excel(df: pd.DataFrame(),sheet_name,file_additional):
    df.to_excel(f'res_{file_additional}.xlsx',
             sheet_name)
    return

def save_dataframes_to_excel(frames_dict, file_name):
    with pd.ExcelWriter(f'res_{file_name}.xlsx') as writer:
        for sheet_name, df in frames_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
def get_top_n_keys_with_max_values(dictionary, n):
   sorted_keys = sorted(dictionary, key=lambda x: dictionary[x], reverse=True)
   return sorted_keys[:n]
           
def make_result(file1,file2,form):
    dicted_file1 = read_docx(file_path = file1, language = 'english')
    dicted_file2 = read_docx(file_path = file2, language = 'english')
    preprocessed_text1 = preprocess_text(text = dicted_file1['text'], language = "english")
    preprocessed_text2 = preprocess_text(text = dicted_file2['text'], language = "english")
    
    main_words1 = get_top_n_keys_with_max_values(calculate_base_bm25(preprocessed_text1.split(" ") ,k1=1.5, b=0.75),round(len(preprocessed_text1.split(" "))*0.05)+1)
    main_words2 = get_top_n_keys_with_max_values(calculate_base_bm25(preprocessed_text2.split(" ") ,k1=1.5, b=0.75),round(len(preprocessed_text2.split(" "))*0.05)+1)

    #list_of_popular_words1 = sorted_keys[-50:]
    
    
    res_frame = pd.DataFrame(columns = ["metric","score","interpretation"])
    stat_frame = pd.DataFrame(columns = ["text","paragraphs_count","words_count","unique_words","sentences_count","names","places","main_words"])
    #str.split(' ')
    
    #Bleu
    bleu_score_value = bleu_score.sentence_bleu([preprocessed_text1.split()], preprocessed_text2.split())
    cur_inter = ""
    if bleu_score_value < 0.75:
        cur_inter = "Тексты существенно различаются"
    elif bleu_score_value >= 0.75 and bleu_score_value <= 0.87:
        cur_inter = "Тексты схожи, но смыслы могут различаться"
    else:
        cur_inter = "Тексты практически или полностью идентичны"
    new_row = {"metric" : "Bleu", 
               "score" : bleu_score_value, 
               "interpretation" : cur_inter
               }
    res_frame = pd.concat([res_frame, pd.DataFrame([new_row])], ignore_index=True)
    
    #Meteor
    meteor_score_value = meteor_score.meteor_score([preprocessed_text1.split()], preprocessed_text2.split())
    if meteor_score_value < 0.80:
        cur_inter = "Тексты существенно различаются"
    elif meteor_score_value >= 0.80 and meteor_score_value <= 0.91:
        cur_inter = "Тексты схожи, но смыслы могут различаться"
    else:
        cur_inter = "Тексты практически или полностью идентичны"
    new_row = {"metric" : "Meteor", 
               "score" : meteor_score_value, 
               "interpretation" : cur_inter
               }
    res_frame = pd.concat([res_frame, pd.DataFrame([new_row])], ignore_index=True)
    
    #Bert
    model_score_value = compare_texts_bert(preprocessed_text1,preprocessed_text2)
    if model_score_value < 0.85:
        cur_inter = "Тексты существенно различаются"
    elif model_score_value >= 0.85 and model_score_value <= 0.93:
        cur_inter = "Тексты схожи, но смыслы могут различаться"
    else:
        cur_inter = "Тексты практически или полностью идентичны"
    new_row = {"metric" : "ML model", 
               "score" : model_score_value, 
               "interpretation" : cur_inter
               }
    res_frame = pd.concat([res_frame, pd.DataFrame([new_row])], ignore_index=True)
    
    """
    return {
        'text': ' '.join(paragraphs),
        'paragraph_count': len(paragraphs),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'unique_word_count': unique_word_count,
        'names': proper_noun_counts,
        'places': place_counts,
        'sentences': sentences
    }

    """
    
    #Statistic
    #df.rename(columns = {' old_col1 ':' new_col1', 'old_col2 ':' new_col2 '}, inplace = True )
    columns_stat_rename = {"text" : "Текст",
                      "paragraphs_count" : "Количество параграфов",
                       "words_count" : "Количество слов",
                       "unique_words" : "Количество уникальных слов",
                       "sentences_count" : "Количество предложений",
                       "names": "Имена собственные",
                       "places": "Места",
                       "main_words":"Ключевые слова"
                       }
    columns_scores_rename = {"metric" : "Инструмент подсчета", 
                               "score" : "Результат оценки", 
                               "interpretation" : "Интерпретация оценки"
                             }
    
    one_row = {"text" : "Текст 1",
               "paragraphs_count" : dicted_file1["paragraph_count"],
               "words_count" : dicted_file1["word_count"],
               "unique_words": dicted_file1["unique_word_count"],
               "sentences_count": dicted_file1["sentence_count"],
               "names": dicted_file1["names"].items(),
               "places": dicted_file1["places"].items(),
               "main_words" : main_words1
               }
    stat_frame = pd.concat([stat_frame, pd.DataFrame([one_row])], ignore_index=True)
    
    one_row = {"text" : "Текст 2",
               "paragraphs_count" : dicted_file2["paragraph_count"],
               "words_count" : dicted_file2["word_count"],
               "unique_words": dicted_file2["unique_word_count"],
               "sentences_count": dicted_file2["sentence_count"],
               "names": dicted_file2["names"].items(),
               "places": dicted_file2["places"].items(),
               "main_words" : main_words2
               }
    
    stat_frame = pd.concat([stat_frame, pd.DataFrame([one_row])], ignore_index=True)
    
    
    stat_frame.rename(columns = columns_stat_rename, inplace = True )
    res_frame.rename(columns = columns_scores_rename, inplace = True )
    
    #Dump
    frames_dict = {
    'Результаты сравнения': res_frame,
    'Характеристики текстов': stat_frame
    }
    
    name_time = str(time.strftime("%Y%m%d-%H%M%S"))
    
    save_dataframes_to_excel(frames_dict, name_time)
    
    #save_dataframe_excel(res_frame,sheet_name='Результаты сравнения',file_additional = name_time)
    #save_dataframe_excel(stat_frame,sheet_name='Характеристики текстов',file_additional = name_time)
    
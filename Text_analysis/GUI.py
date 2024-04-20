import PySimpleGUI as sg
from nltk.translate import bleu_score
from nltk.translate import meteor_score
from docx import Document
import docx
from Models.Bert import *
from Preprocessing.preprocess_text import *
"""
def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text
"""

main_layout = [
    [sg.TabGroup([[
        sg.Tab('Выбрать инструмент', [
            [sg.Text('Выберите метрику:')],
            [sg.Radio('BLEU', 'metric', default=True, key='bleu_metric'), sg.Radio('METEOR', 'metric', key='meteor_metric'), sg.Radio('Model', 'metric', key='model_metric')],
            [sg.Button('Next')]
        ]),
        sg.Tab('Выбрать файлы', [
            [sg.Text('Выберите первый текст:'), sg.InputText(key='file1'), sg.FileBrowse()],
            [sg.Radio('ENGLISH', group_id = 'lang1', default=True, key='file1_english'), sg.Radio('SPAIN', group_id = 'lang1', key='file1_spain'), sg.Radio('RUSSIAN', group_id = "lang1", key='file1_russian')],
            [sg.Text("", size=(80, 1), key='-OUTPUT1-')],
            [sg.Text('Выберите второй текст:'), sg.InputText(key='file2'), sg.FileBrowse()],
            [sg.Radio('ENGLISH', group_id = 'lang2', default=True, key='file2_english'), sg.Radio('SPAIN', group_id = 'lang2', key='file2_spain'), sg.Radio('RUSSIAN', group_id = "lang2", key='file2_russian')],
            [sg.Text("", size=(80, 1), key='-OUTPUT2-')],
            [sg.Button('Сравнить')],
            [sg.Output(size=(80, 10))]
        ], key='_TAB_FILES_')
    ]])]
]

sg.theme("LightGreen")
window = sg.Window('Анализ и сравнение художественных текстов', main_layout,icon='Media/main.ico').finalize()
#window.Maximize()


while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break

    if event == 'Next':
        window['_TAB_FILES_'].select()

    if event == 'Сравнить':
        try:
            """
            {
                'text': ' '.join(paragraphs),
                'paragraph_count': len(paragraphs),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'proper_nouns': proper_noun_counts,
                'sentences': sentences
            }
            """
            dicted_file1 = read_docx(file_path = values['file1'], language = 'english')
            dicted_file2 = read_docx(file_path = values['file2'], language = 'english')
            window['-OUTPUT1-'].update(f"Количество параграфов: {dicted_file1['paragraph_count']} Количество слов: {dicted_file1['word_count']} Имён собственных: {len(dicted_file1['proper_nouns'])}")
            preprocessed_text1 = preprocess_text(text = dicted_file1['text'], language = "english")
            preprocessed_text2 = preprocess_text(text = dicted_file2['text'], language = "english")
            window['-OUTPUT2-'].update(f"Количество параграфов: {dicted_file2['paragraph_count']} Количество слов: {dicted_file2['word_count']} Имён собственных: {len(dicted_file2['proper_nouns'])}")
            
        except Exception as e:
            print("Произошла ошибка: ",str(e))
        
        #print(values[""])
        #dict_text2 = read_docx(file_path = values['file2'], language = 'english')
        #print(res)
        
        if values['bleu_metric']:
            bleu_score_value = bleu_score.sentence_bleu([preprocessed_text1.split()], preprocessed_text2.split())
            print(f"Оценка схожести по BLEU: {bleu_score_value}")
        elif values['meteor_metric']:
            meteor_score_value = meteor_score.meteor_score([preprocessed_text1.split()], preprocessed_text2.split())
            print(f"Оценка схожести по METEOR: {meteor_score_value}")
        elif values['model_metric']:
            model_score_value = compare_texts_bert(preprocessed_text1,preprocessed_text2)
            print(f"Оценка схожести согласно модели: {model_score_value}")

window.close()
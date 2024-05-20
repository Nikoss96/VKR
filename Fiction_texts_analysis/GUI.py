import PySimpleGUI as sg
from nltk.translate import bleu_score
from nltk.translate import meteor_score
from docx import Document
import docx
from Models.Bert import *
from Preprocessing.preprocess_text import *
from Preprocessing.window_changes import *
from Preprocessing.save import *
from Metrics.Nist import *
from Models.xlnet import *
"""
def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text
"""
button_image_path = 'Media/button_brwon.png'

main_layout = [
    [sg.TabGroup(
        [
            [
                
        sg.Tab('Выбрать инструмент', [
            [sg.Text('Выберите метрику:')],
            [sg.Radio('BLEU', 'metric', default=True, key='bleu_metric'), 
             sg.Radio('METEOR', 'metric', key='meteor_metric'), 
             sg.Radio('Bert', 'metric', key='bert_metric'),
             sg.Radio('XLnet', 'metric', key='xlnet_metric'),
             sg.Radio('L12', 'metric', key='l12_metric'),
             sg.Radio('Mpnet', 'metric', key='mpnet_metric'),
             sg.Radio('Использовать все', 'metric', key='all_metric')],
            [sg.Text('Дополнительные возможности')],
            [sg.Radio('Эмоциональная оценка', 'emotional', default=False, key='emotional_evaluation')],
            [sg.Button('Далее')]
        ]),
        
        sg.Tab('Выбрать файлы', [
            [sg.Text('Выберите первый текст:'), sg.InputText(key='file1'), sg.FileBrowse("Выбрать")],
            [sg.Radio('ENGLISH', group_id = 'lang1', default=True, key='file1_english'), sg.Radio('SPAIN', group_id = 'lang1', key='file1_spain'), sg.Radio('RUSSIAN', group_id = "lang1", key='file1_russian')],
            [sg.Text("", size=(80, 1), key='-OUTPUT1-')],
            [sg.Text('Выберите второй текст:'), sg.InputText(key='file2'), sg.FileBrowse("Выбрать")],
            [sg.Radio('ENGLISH', group_id = 'lang2', default=True, key='file2_english'), sg.Radio('SPAIN', group_id = 'lang2', key='file2_spain'), sg.Radio('RUSSIAN', group_id = "lang2", key='file2_russian')],
            [sg.Text("", size=(80, 1), key='-OUTPUT2-')],
            [sg.Button('Сравнить')],
            [sg.Output(size=(80, 10))]
        ], 
        key='_TAB_FILES_'),
        
       sg.Tab('Экспорт результатов', [
           [sg.Text('Выберите вариант:')],
           [sg.Radio('Excel', 'export', default=True, key='excel_export'), sg.Radio('Text', 'export', key='text_export')],
           [sg.Button('Экспортировать')]
       ]),  
    ]
    ]
    )
        ]
]
# DarkBrown6 LightGrey6 Reddit 
sg.theme("Reddit")

window = sg.Window('Анализ и сравнение художественных текстов', main_layout,icon='Media/main.ico').finalize()
#window.Maximize()


while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break

    if event == 'Далее':
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
            window['-OUTPUT1-'].update(f"Количество параграфов: {dicted_file1['paragraph_count']} Количество слов: {dicted_file1['word_count']} Имён собственных: {len(dicted_file1['names'])}")
            preprocessed_text1 = preprocess_text(text = dicted_file1['text'], language = "english")
            preprocessed_text2 = preprocess_text(text = dicted_file2['text'], language = "english")
            window['-OUTPUT2-'].update(f"Количество параграфов: {dicted_file2['paragraph_count']} Количество слов: {dicted_file2['word_count']} Имён собственных: {len(dicted_file2['names'])}")
            
        except Exception as e:
            print("Произошла ошибка: ",str(e))
    
        if values['bleu_metric']:
            bleu_score_value = bleu_score.sentence_bleu([preprocessed_text1.split()], preprocessed_text2.split())
            print(f"Оценка схожести по BLEU: {bleu_score_value}")
        elif values['meteor_metric']:
            meteor_score_value = meteor_score.meteor_score([preprocessed_text1.split()], preprocessed_text2.split())
            print(f"Оценка схожести по METEOR: {meteor_score_value}")
        elif values['bert_metric']:
            bert_score_value = compare_texts_bert(preprocessed_text1,preprocessed_text2)
            print(f"Оценка схожести согласно Bert: {bert_score_value}")
        elif values['xlnet_metric']:
            _,xlnet_score_value = compare_texts_xlnet(preprocessed_text1,preprocessed_text2)
            print(f"Оценка схожести согласно XLnet: {xlnet_score_value}")

            
            
            
    if event == 'Экспортировать':
        if values['excel_export']:
            make_result(values['file1'],values['file2'],form = 'excel_export')   
        elif values['text_export']:
            pass
window.close()
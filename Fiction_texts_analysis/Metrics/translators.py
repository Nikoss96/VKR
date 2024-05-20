import docx 
import nltk
import re
import googletrans
from googletrans import Translator
from nltk.translate.bleu_score import corpus_bleu
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import requests
from nltk.translate.bleu_score import sentence_bleu
from Preprocessing.preprocess_text import *
from nltk.translate import bleu_score

#from google_trans_new import google_translator
"""
translator = Translator() 
def cut_text(textik:str):
    cur = re.findall('[A-Za-zА-ЯЁа-яё]+-[A-Za-zА-ЯЁа-яё]+|[A-Za-zА-ЯЁа-яё]+', textik)
    fin = cur[:4999]
    return ' '.join(fin)
from nltk.translate.bleu_score import sentence_bleu
def read_docx(file_path):
    doc = docx.Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return ' '.join(text)
def preprocess_text(text,language):
    stopwords_list = set(stopwords.words(language))
    tokens = nltk.word_tokenize(text,language=language)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    filtered_text = [word for word in tokens if word.lower() not in stopwords_list]
    stemmed_words = [stemmer.stem(word) for word in filtered_text]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]
    #print(' '.join(lemmatized_words))
    return ' '.join(lemmatized_words)
#Pathes for docs
path_eng = '..\\Texts\\1_Blow-Up.docx'
path_spain = '..\\Texts\\1_Las_babas_del_diablo.docx'
path_eng_internet = '..\\Texts\\Blow_internet.docx'

eng_parse = read_docx(path_eng)
spain_parse = read_docx(path_spain)
words_spain = spain_parse
result_spain = translator.translate(str(spain_parse),dest = 'en')
#print(result_spain.src)
#print(result_spain.dest)
#print(result_spain.origin)
#print(result_spain.text)
#print(result_spain.pronunciation)
words_eng = cut_text(eng_parse)
result_eng = words_eng[:len(result_spain.text)]

internet_eng_parse = read_docx(path_eng_internet)
words_eng_internet = cut_text(internet_eng_parse)
result_eng_internet = words_eng[:len(result_spain.text)]
#filtered_eng_words = [token for token in words_eng if token not in nltk.corpus.stopwords.words('english')]
#print(googletrans.LANGUAGES)

def calc_metrics(text_eng:list,text_spain:str):
    bleu_score = sentence_bleu(text_eng, text_spain)
    #print(text_eng,'\n')
    #print(text_spain,'\n')
    print("BLEU Score:", bleu_score)

#preprocess_text(eng_parse)
#calc_metrics(words_eng[:len(result_spain.text)],result_spain.text)
#calc_metrics([preprocess_text(text = spain_parse,language = 'spanish')],preprocess_text(internet_eng_parse,language = 'english'))
into_eng = re.findall('[A-Za-zА-ЯЁа-яё]+-[A-Za-zА-ЯЁа-яё]+|[A-Za-zА-ЯЁа-яё]+', result_eng)
into_spain = re.findall('[A-Za-zА-ЯЁа-яё]+-[A-Za-zА-ЯЁа-яё]+|[A-Za-zА-ЯЁа-яё]+', result_spain.text)
into_eng_internet = re.findall('[A-Za-zА-ЯЁа-яё]+-[A-Za-zА-ЯЁа-яё]+|[A-Za-zА-ЯЁа-яё]+', result_eng_internet)
into_eng_last = list(into_eng[:min(len(into_spain),len(into_eng))])
into_spain_last = list(into_spain[:min(len(into_spain),len(into_eng))])
into_eng_last_internet = list(into_eng_internet[:min(len(into_eng_internet),len(into_eng))])
bleu1 = corpus_bleu(into_eng_last,into_spain_last, weights=(1.0, 0, 0, 0))
print(f'Схожесть после перевода через гугл: {bleu1}')
into_eng_last = list(into_eng[:min(len(into_eng_internet),len(into_eng))])
into_eng_last_internet = list(into_eng_internet[:min(len(into_eng_internet),len(into_eng))])
bleu2 = corpus_bleu(into_eng_last,into_eng_last_internet, weights=(1.0, 0, 0, 0))
print(f'Схожесть после перевода человеком: {bleu2}')
"""


### Yandex compare
def yandex_compare(path_eng,path_translated ,path_eng_internet):
    path_eng = '..\\Texts\\1_Blow-Up.docx'
    path_translated = '..\\Texts\\1_Yandex.docx'
    path_eng_internet = '..\\Texts\\Blow_internet.docx'
    
    
    dicted_file_orig = read_docx(file_path = path_eng, language = 'english')
    dicted_file_translator = read_docx(file_path = path_translated, language = 'english')
    dicted_file_internet = read_docx(file_path = path_eng_internet, language = 'english')
    
    preprocessed_text_orig = preprocess_text(text = dicted_file_orig['text'], language = "english")
    preprocessed_text_translator = preprocess_text(text = dicted_file_translator['text'], language = "english")
    preprocessed_text_internet = preprocess_text(text = dicted_file_internet['text'], language = "english")
    
    bleu1 = bleu_score.sentence_bleu([preprocessed_text_orig.split()], preprocessed_text_translator.split())
    #bleu1 = corpus_bleu([preprocessed_text_orig.split()],[preprocessed_text_translator.split()], weights=(1.0, 0, 0, 0))
    print(f'Схожесть после перевода через Yandex: {bleu1}')
    bleu2 = bleu_score.sentence_bleu([preprocessed_text_orig.split()], preprocessed_text_internet.split())
    #bleu2 = corpus_bleu([preprocessed_text_orig.split()],[preprocessed_text_internet.split()], weights=(1.0, 0, 0, 0))
    print(f'Схожесть после перевода человеком: {bleu2}')
### Google compare






"""
LANGUAGES = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'he': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'or': 'odia',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'ug': 'uyghur',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
"""





def google_compare(path_eng, path_spain, path_eng_internet):  
 path_eng = path_eng
 path_spain = path_spain
 path_eng_internet = path_eng_internet
 
 
 dicted_file_orig = read_docx(file_path = path_eng, language = 'english')
 dicted_file_spain = read_docx(file_path = path_spain, language = 'spanish')
 dicted_file_internet = read_docx(file_path = path_eng_internet, language = 'english')
 

 check = "Nunca se sabrá cómo hay que contar esto, si en primera persona o en segunda, usando la tercera del plural o inventando continuamente formas que no servirán de nada. "
 translator = Translator() 
 result_spain = translator.translate(dicted_file_spain["text"][:5000],src = 'es',dest = 'en')
 
 #print(result_spain) #result_spain.text
 
 preprocessed_text_orig = preprocess_text(text = dicted_file_orig['text'][:5000], language = "english")
 preprocessed_text_translator = preprocess_text(text = result_spain.text, language = "english")
 preprocessed_text_internet = preprocess_text(text = dicted_file_internet['text'][:5000], language = "english")
 
 bleu1 = bleu_score.sentence_bleu([preprocessed_text_orig.split()], preprocessed_text_translator.split())
 #bleu1 = corpus_bleu([preprocessed_text_orig.split()],[preprocessed_text_translator.split()], weights=(1.0, 0, 0, 0))
 print(f'Схожесть после перевода через Google: {bleu1}')
 bleu2 = bleu_score.sentence_bleu([preprocessed_text_orig.split()], preprocessed_text_internet.split())
 #bleu2 = corpus_bleu([preprocessed_text_orig.split()],[preprocessed_text_internet.split()], weights=(1.0, 0, 0, 0))
 print(f'Схожесть после перевода человеком: {bleu2}')
 
google_compare(path_eng = '..\\Texts\\1_Blow-Up.docx',path_spain = '..\\Texts\\1_Las_babas_del_diablo.docx',path_eng_internet = '..\\Texts\\Blow_internet.docx')
yandex_compare(path_eng = '..\\Texts\\1_Blow-Up.docx',path_translated = '..\\Texts\\1_Yandex.docx',path_eng_internet = '..\\Texts\\Blow_internet.docx')
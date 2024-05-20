import docx 
import nltk
import re
import googletrans
from googletrans import Translator
from nltk.translate.bleu_score import corpus_bleu
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
#from google_trans_new import google_translator

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
path_eng = 'Texts\\1_Blow-Up.docx'
path_spain = 'Texts\\1_Las_babas_del_diablo.docx'
path_eng_internet = 'Texts\\Blow_internet.docx'

eng_parse = read_docx(path_eng)
spain_parse = read_docx(path_spain)
words_spain = cut_text(spain_parse)
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
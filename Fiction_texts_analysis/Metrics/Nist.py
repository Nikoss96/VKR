from nltk.translate import nist_score
import nltk

def calc_nist_score(text_1,text_2):
    tokens1 = nltk.word_tokenize(text_1.lower())
    tokens2 = nltk.word_tokenize(text_2.lower())
    
    nist_score = nltk.translate.nist_score.corpus_nist([tokens1], [tokens2], n=5)
    
    return nist_score
    #score = nist_score.sentence_nist([text_1], [text_2])
    #return score
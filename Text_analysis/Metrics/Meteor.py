from nltk.translate import meteor_score
import os
import sys
from Preprocessing.preprocess_text import read_docx,preprocess_text

def check_path_extensions(path,extensions):
    extensions = set(extensions)
    for root1, dirs1, file1 in os.walk(os.path.abspath(path)):
        for file in file1:
            _, extension = os.path.splitext(file)
            
            if extension not in extensions:
                print("Один из файлов в директории имеет неподдерживаемое расширение.")
                sys.exit()
         
def get_meteor_score(main_path,test_path,frame):
    check_path_extensions(path = main_path, extensions = [".docx"])
    check_path_extensions(path = test_path, extensions = [".docx"])

    for cur_main_path,_,cur_main_file in os.walk(os.path.abspath(main_path)):
        for file_main in cur_main_file:
            for cur_test_path,_,cur_test_file in os.walk(os.path.abspath(test_path)):
                for file_test in cur_test_file:
                    print()
                    dicted_file1 = read_docx(file_path = cur_main_path + "\\" + file_main, language = 'english')
                    dicted_file2 = read_docx(file_path = cur_test_path + "\\" + file_test, language = 'english')
                    preprocessed_text1 = preprocess_text(text = dicted_file1['text'], language = "english")
                    preprocessed_text2 = preprocess_text(text = dicted_file2['text'], language = "english")
                    
                    meteor_score_value = meteor_score.meteor_score([preprocessed_text1.split()], preprocessed_text2.split())
                    print(meteor_score_value)
                #return meteor_score_value

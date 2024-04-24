import os
import sys
import collections
import pandas as pd
def check_path(path):
    if os.path.isdir(path):
        return True
    else:
        return False
    
def get_path(text):
    file_path = str(input(text+"\n"))
    if check_path(file_path):
        return file_path
    else:
        print("Директории не найдено.")
        sys.exit()
    
    
def check_instruments(instruments):
    instruments_cur = set(["meteor","bleu","all-minilm-l6","bert","all-minilm-l6","xlnet","bge-m3","all"])
    for instrument in instruments:
        if instrument not in instruments_cur:
            
            return False
        else:
            pass
    return True

def get_instruments():
    print("Выберите инструмент анализа.")
    print("Доступные инструменты (Вводить через пробел!): \n - meteor \n - bleu \n - bert \n - all-minilm-l6 \n - bge-m3 \n - xlnet \n - all")
    choosen_instruments = str(input())
    if check_instruments(choosen_instruments.split()):
        return choosen_instruments
    else:
        print('Присутствует некорректный инструмент.')
        sys.exit()

def get_dir_names(main_path):
    main_names = []
    main_names += os.listdir(main_path)
    for i in range(len(main_names)):
        name, _ = os.path.splitext(main_names[i])
        main_names[i] = name
    return main_names

def check_path_extensions(path,extensions):
    extensions = set(extensions)
    for root1, dirs1, file1 in os.walk(os.path.abspath(path)):
        for file in file1:
            _, extension = os.path.splitext(file)
            if extension not in extensions:
                print("Один из файлов в директории имеет неподдерживаемое расширение.")
                sys.exit()
    
def main():
    
    main_path = get_path("Укажите папку с эталонными файлами.")
    test_path = get_path("Укажите папку с тестовыми переводами.")
    instruments = get_instruments().split()
    
    dir_names1 = get_dir_names(main_path)
    dir_names2 = get_dir_names(test_path)
    
    columns = ['metric'] + [str(a) + " сравнили с " + str(b) for a in dir_names1 for b in dir_names2]
    result = pd.DataFrame(columns)
    print(result)
    
    """
    if "all" in set(instruments):
        print("Ебанутый, мне лень!")
    else:
        for instrument in instruments:
            if instrument == "meteor":
                get_bert_score(main_path,text_path)  
            elif instrument == "bleu":
                get_bleu_score(main_path,text_path) 
    """
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()
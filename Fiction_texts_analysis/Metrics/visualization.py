import matplotlib.pyplot as plt
def plot_2_dicts(dict1,dict2,axisx,axisy):
    plt.figure()
    for key in dict1.keys():
        x1= dict1[key]
        y1 = dict2[key]
        plt.scatter(x1, y1, color='blue')
        plt.text(x1, y1, key, fontsize=10, ha='center', va='bottom')

    plt.xlabel(axisx)
    plt.ylabel(axisy)
    plt.title('Общий график')
    plt.show()
    
def normalize_dict_values(dictionary):
    normalized_dict = {}
    dict_values = list(dictionary.values())
    min_value, max_value = min(dict_values), max(dict_values)
    for key in dictionary.keys():
        #print(key)
        normalized_value = (dictionary[key] - min_value) / (max_value - min_value)
        #print(1 - normalized_value)
        normalized_dict[key] = 1 - normalized_value
    
    return normalized_dict
# Заданные данные: массив с парами название - оценка
max_metric = {
    'bge-m3': 0.8371274638829285,
    'mdeberta-v3-base': 0.3104035938599628,
    'roberta-large-nli-stsb-mean-tokens': 0.8918350910524713,
    'all-MiniLM-L6-v2': 0.8135998636801557,
    'all-MiniLM-L12-v2': 0.8289241529037255,
    'all-mpnet-base-v2': 0.8328013200122332,
    'distiluse-base-multilingual-cased-v2': 0.7935237232045801,
    'paraphrase-albert-base-v2': 0.8041280611805134,
    'paraphrase-multilingual-MiniLM-L12-v2': 0.846953746993021,
    'xlnet-large-cased': 0.876953746993021,
}

time = {
        'bge-m3': 611.69,
        'mdeberta-v3-base': 267.54,
        'roberta-large-nli-stsb-mean-tokens': 661.31,
        'all-MiniLM-L6-v2': 25.47,
        'all-MiniLM-L12-v2': 59.72,
        'all-mpnet-base-v2': 133.83,
        'distiluse-base-multilingual-cased-v2': 72.1,
        'paraphrase-albert-base-v2': 213.1,
        'paraphrase-multilingual-MiniLM-L12-v2': 63.54,
        'xlnet-large-cased': 33.54,
}

weighted_average = {
    'bge-m3': 0.8371274638829285,
    'mdeberta-v3-base': 0.3104035938599628,
    'roberta-large-nli-stsb-mean-tokens': 0.8918350910524713,
    'all-MiniLM-L6-v2': 0.8135998636801557,
    'all-MiniLM-L12-v2': 0.8289241529037255,
    'all-mpnet-base-v2': 0.8328013200122332,
    'distiluse-base-multilingual-cased-v2': 0.7935237232045801,
    'paraphrase-albert-base-v2': 0.8041280611805134,
    'paraphrase-multilingual-MiniLM-L12-v2': 0.846953746993021,
    'xlnet-large-cased': 0.876953746993021,
}

w1 = 0.1
w2 = 0.9
time_normalized = normalize_dict_values(time)
for i in max_metric.keys():
    weighted_average[i] = time_normalized[i] * w1 + max_metric[i] * w2 
evaluation = [max_metric,time]




names = list(max_metric.keys())
values = list(max_metric.values())

plt.barh(range(len(max_metric)), values, align='center', color='skyblue')
plt.yticks(range(len(max_metric)), names)
"""
for i in range(len(names)):
    plt.text(i, values[i] - 0.5, names[i], ha='center',rotation=90)
"""

plt.xlabel('Оценки (MTEB)')
plt.ylabel('Модели')
plt.title('Гистограмма оценок моделей')
plt.show()





names = list(time.keys())
values = list(time.values())

plt.barh(range(len(time)), values, align='center', color='skyblue')
plt.yticks(range(len(time)), names)

"""
for i in range(len(names)):
    plt.text(i, values[i] - 0.5, names[i], ha='center',rotation=90)
"""

plt.xlabel('Время выполнения (секунды)')
plt.ylabel('Модели')
plt.title('Гистограмма оценок моделей')
plt.show()






names = list(weighted_average.keys())
values = list(weighted_average.values())

plt.barh(range(len(weighted_average)), values, align='center', color='skyblue')
plt.yticks(range(len(weighted_average)), names)

"""
for i in range(len(names)):
    plt.text(i, values[i] - 0.5, names[i], ha='center',rotation=90)
"""

plt.xlabel('Взвешенная оценка')
plt.ylabel('Модели')
plt.title('Гистограмма оценок моделей')
plt.show()


plot_2_dicts(dict1 = max_metric, dict2 = time ,axisx = "Метрика", axisy = "Время выполнения") 
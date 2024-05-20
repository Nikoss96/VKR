import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from nltk.translate import bleu_score
from nltk.translate import meteor_score
from docx import Document
# Импортируйте ваши собственные модули
from Models.Bert import compare_texts_bert
from Preprocessing.preprocess_text import preprocess_text
import docx

def read_docx(file_path, language):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

def choose_file(entry_widget):
    file_path = filedialog.askopenfilename()
    if file_path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)

def compare_texts():
    try:
        file1 = entry_file1.get()
        file2 = entry_file2.get()
        text1 = preprocess_text(text=read_docx(file1, 'english'), language='english')
        text2 = preprocess_text(text=read_docx(file2, 'english'), language='english')
        
        result_text = "No valid metric selected"
        if var_metric.get() == 1:  # BLEU
            score = bleu_score.sentence_bleu([text1], text2)
            result_text = f"BLEU Score: {score:.4f}"
        elif var_metric.get() == 2:  # METEOR
            score = meteor_score.meteor_score([text1], text2)
            result_text = f"METEOR Score: {score:.4f}"
        elif var_metric.get() == 3:  # Model
            score = compare_texts_bert(text1, text2)
            result_text = f"Model Score: {score:.4f}"
        
        output_text_widget.delete(1.0, tk.END)
        output_text_widget.insert(tk.END, result_text)

    except Exception as e:
        output_text_widget.delete(1.0, tk.END)
        output_text_widget.insert(tk.END, f"Error: {str(e)}")

root = tk.Tk()
root.title("Анализ и сравнение художественных текстов")

tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Выбрать инструмент')
tab_control.add(tab2, text='Выбрать файлы')
tab_control.add(tab3, text='Экспорт результатов')
tab_control.pack(expand=1, fill='both', padx=10, pady=10)

metric_frame = ttk.LabelFrame(tab1, text='Metrics')
metric_frame.pack(fill='both', expand=1, padx=10, pady=10)
var_metric = tk.IntVar(value=1)
metrics = [('BLEU', 1), ('METEOR', 2), ('Model', 3)]
for text, value in metrics:
    ttk.Radiobutton(metric_frame, text=text, variable=var_metric, value=value).pack(anchor='w', pady=5)

file_frame = ttk.Frame(tab2)
file_frame.pack(fill='both', expand=1, padx=10, pady=10)

ttk.Label(file_frame, text='Первый текст:').pack()
entry_file1 = ttk.Entry(file_frame)
entry_file1.pack(fill='x', padx=5, pady=5)
ttk.Button(file_frame, text='Browse', command=lambda: choose_file(entry_file1)).pack(fill='x', padx=5, pady=5)

ttk.Label(file_frame, text='Второй текст:').pack()
entry_file2 = ttk.Entry(file_frame)
entry_file2.pack(fill='x', padx=5, pady=5)
ttk.Button(file_frame, text='Browse', command=lambda: choose_file(entry_file2)).pack(fill='x', padx=5, pady=5)

ttk.Button(file_frame, text='Сравнить', command=compare_texts).pack(pady=10)

output_text_widget = ScrolledText(file_frame, wrap=tk.WORD)
output_text_widget.pack(fill='both', expand=1, padx=5, pady=5)

export_frame = ttk.LabelFrame(tab3, text='Export Options')
export_frame.pack(fill='both', expand=1, padx=10, pady=10)
var_export = tk.IntVar(value=1)
exports = [('Excel', 1), ('Text', 2)]
for text, value in exports:
    ttk.Radiobutton(export_frame, text=text, variable=var_export, value=value).pack(anchor='w', pady=5)

root.mainloop()
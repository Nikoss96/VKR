from mteb import MTEB
from sentence_transformers import SentenceTransformer
#from collection import defaultdict
import pandas as pd
import tqdm

data = "STS12"
def evaluate_model(model_name):
    #model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    evaluation = MTEB(tasks=[data])
    results = tqdm(evaluation.run(model, output_folder=f"results/{model_name}"))
    
    return results



def main():
    frame = pd.DataFrame(columns = ["model","cos_sim","manhattan","euclidean","time"])
    res = dict()
    #"sentence-transformers/all-MiniLM-L6-v2","roberta-large-nli-stsb-mean-tokens",'sentence-transformers/all-MiniLM-L12-v2','xlnet-large-cased','BAAI/bge-m3','sentence-transformers/all-mpnet-base-v2','sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2','sentence-transformers/paraphrase-albert-base-v2','sentence-transformers/distiluse-base-multilingual-cased-v2','microsoft/mdeberta-v3-base'
    models = ['microsoft/mdeberta-v3-base']
    for i in models:
        try:
            cur = evaluate_model(i)
            frame.loc[len(frame.index)] = [i,cur[data]["test"]["cos_sim"]["pearson"], cur[data]["test"]["manhattan"]["pearson"], cur[data]["test"]["euclidean"]["pearson"],cur[data]["test"]["evaluation_time"]] 
        except Exception as e:
            print(e)
    print(frame)
           


if __name__ == "__main__":
    main()
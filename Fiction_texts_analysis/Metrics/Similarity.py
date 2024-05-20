from scipy.spatial.distance import cosine

def get_cosine_similarity(embedding1,embedding2):
    res = 1 - cosine(embedding1, embedding2)
    return res
        
    


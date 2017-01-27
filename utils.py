import codecs
import numpy as np

def build_word_vector_matrix(vector_file, n_words=None):
    '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays
       Example Usage:
            embeddings, labels, id_to_word, word_to_id = build_word_vector_matrix('vectors_d50.txt')
    '''
    
    np_arrays = []
    labels = []

    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            sr = line.split()
            labels.append(sr[0])
            np_arrays.append(np.array([float(j) for j in sr[1:]]))
            if n_words and i == n_words - 1:
                break
    embeddings = np.array(np_arrays)
    id_to_word = dict(zip(range(len(labels)), labels))
    word_to_id = dict((v,k) for k,v in id_to_word.items())
    return embeddings, labels, id_to_word, word_to_id

def move_to_glove(move, embeddings, word_to_id):
    if move in word_to_id:
        return embeddings[word_to_id[move]]
    else :
        # return the "I've never seen that" vector
        return np.full(len(embeddings[0]), 1, dtype=np.float32) 

def encode_moves(moves, length):   
    encoded = [move_to_glove(m) for m in moves]
    while len(encoded) < length:
        encoded.append(np.zeros(length))
    return encoded
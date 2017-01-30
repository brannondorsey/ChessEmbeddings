
# coding: utf-8

# ## Imports

# In[7]:

import sys, pdb
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM

# Add the parent directory to PYTHONPATH so that we can use utils.py
sys.path.append('../..')
import utils

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ## Load Vector Embeddings

# In[8]:

d = 50 #dimensionality of word vectors
data = utils.build_word_vector_matrix('../data/embeddings/5/vectors_d{}.txt'.format(d))
embeddings, labels, id_to_move, move_to_id = data


# ## Load Move Dataset

# In[9]:

with open('../data/test_moves.txt', 'r') as f:
    moves = f.read().split()
    
# reduce number of moves for now
moves = moves[:10000]
uniq_moves = list(set(moves))
print('{} unique moves in vector encoding'.format(len(labels)))
print('{} unique moves in training set'.format(len(uniq_moves)))


# In[10]:

window_size = 20
step = 1
input_moves = []
output_moves = []

for i in range(0, len(moves) - window_size, step):
    input_moves.append(moves[i:i + window_size])
    output_moves.append(moves[i + window_size])

output_move_ids = []
for m in output_moves:
    if m in move_to_id:
        output_move_ids.append(move_to_id[m])
    else:
        # unknown
        output_move_ids.append(-1)

y = to_categorical(output_move_ids, len(labels))
X = []
print('{} input sequences'.format(len(input_moves)))
unknown_moves = set()
for sequence in input_moves:
    seq = []
    for move in sequence:
        if move in move_to_id:
            #vec is of length 50
            vec = embeddings[move_to_id[move]]
            seq.append(vec.tolist())
        else:
            unknown_moves.add(move)
            #this zero array is of length 50
            seq.append(np.zeros((d,)).tolist())
    X.append(seq)

# really wierd fucking bug right here
w = np.array(X)
X = w

print('Moves not found in vector embedding dictionary:')
print(*unknown_moves)


# In[11]:

model = Sequential()
model.add(LSTM(1024, return_sequences=True, batch_input_shape=(None, window_size, d)))
model.add(Dropout(0.5))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(len(labels)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, decay=0.0001), metrics=['accuracy'])
print(model.summary())


# In[12]:

# more epochs is usually better, but training can be very slow if not on a GPU
epochs = 30
history = model.fit(X, y, batch_size=32, nb_epoch=epochs, validation_split=0.2, verbose=1)


# In[13]:

utils.plot_model_results(history)   


# In[15]:

predicted = model.predict(X)
ids = [np.argmax(p) for p in predicted]
moves = [id_to_move[i] for i in ids]


# In[16]:

moves


# In[ ]:




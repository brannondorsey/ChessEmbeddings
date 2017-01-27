import codecs, json, os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint

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

def create_model(num_inputs, num_outputs, verbose=False, save_dir=None, hyper=None):
    
    callbacks = []
    if not hyper:
        hyper = {
            'layers': [{
                'units': 1024,
                'init': 'normal',
                'activation': 'relu',
                'dropout': 0.5
                },{
                'units': 1024,
                'init': 'normal',
                'activation': 'relu',
                'dropout': 0.5

                },{
                'units': 1024,
                'init': 'normal',
                'activation': 'relu',
                'dropout': 0.5
                },
                {
                'units': num_outputs,
                'init': 'normal',
                'activation': 'softmax'
            }],
            'loss': 'categorical_crossentropy',
            'optimizer': {
                'type': 'rmsprop',
                'lr': 0.001
            },
            'metrics': ['accuracy']

        }
    
    if save_dir:
        # create save_dir if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
       
        # save hyperparameters
        with open('{}/hyperparameters.json'.format(save_dir), 'w') as f:
            json.dump(hyper, f, indent=4, sort_keys=True)
            
        # create checkpoints dir if it doesn't exist
        checkpoint_dir = '{}/checkpoints'.format(save_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        filepath = checkpoint_dir + '/checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5'
        callbacks.append(ModelCheckpoint(filepath, 
                                         monitor='val_acc', 
                                         verbose=1, 
                                         save_best_only=True, 
                                         mode='max'))
    
    model = Sequential()
    
    # layers
    for i, layer in enumerate(hyper['layers']):
        
        if verbose:
                print('Adding layer:')
                print('     units:      {}'.format(layer['units']))
                print('     init:       {}'.format(layer['init']))
                print('     activation: {}'.format(layer['activation']))
                
        # add layer
        if i == 0:
            
            if verbose:
                print('     input_dim:  {}'.format(num_inputs))
            
            model.add(Dense(layer['units'], 
                            input_dim=num_inputs, 
                            init=layer['init'], 
                            activation=layer['activation']))
        else:
            model.add(Dense(layer['units'], 
                            init=layer['init'], 
                            activation=layer['activation']))
        # add dropout
        if 'dropout' in layer:
            if verbose:
                print('     dropout:    {:.2f}'.format(layer['dropout']))
            model.add(Dropout(layer['dropout']))
    
    if verbose:
        print('optimizer:     {}'.format(hyper['optimizer']['type']))
        print('learning rate: {}'.format(hyper['optimizer']['lr']))
    
    optim = hyper['optimizer']['type']
    if optim == 'rmsprop':
        optimizer = RMSprop(lr=hyper['optimizer']['lr'])
    elif optim == 'sgd':
        optimizer = SGD(lr=hyper['optimizer']['lr'])
    elif optim == 'adam':
        optimizer = Adam(lr=hyper['optimizer']['lr'])
    else:
        raise Exception('Unsupported optimization algorithm')
    
    if verbose:
        print('loss:    {}'.format(hyper['loss']))
        print('metrics: {}'.format(*hyper['metrics']))
    
    model.compile(loss=hyper['loss'], optimizer=optimizer, metrics=hyper['metrics'])
    return model, callbacks
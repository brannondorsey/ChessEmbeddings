import codecs, json, os, glob, pdb
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing 
import matplotlib.pyplot as plt

def preprocess(X, y):
    # shape = X.shape
    # X = np.reshape(X, (shape[0], shape[1] * shape[2]))
    # X = preprocessing.scale(X, axis=0)
    # X = np.reshape(X, shape)

    # rng_state = np.random.get_state()
    # np.random.shuffle(X)
    # np.random.set_state(rng_state)
    # np.random.shuffle(y)

    return X, y

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

def plot_model_results(history, save_dir=None):
    
    plt.rcParams["figure.figsize"] = (12, 8)
    
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.show()
    
    argmax = np.argmax(history.history['val_acc']) 
    mess = 'Highest val_acc at epoch {} with value of {:.3f}'
    print(mess.format(argmax + 1, history.history['val_acc'][argmax]))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.show()
    
    mess = 'Lowest val_loss at epoch {} with value of {:.2f}'
    argmin = np.argmin(history.history['val_loss'])
    print(mess.format(argmin + 1, history.history['val_loss'][argmin]))

    if 'fbeta_score' in history.history:
        mess = 'Lowest val_loss at epoch {} with value of {:.2f}'
        argmin = np.argmin(history.history['fbeta_score'])
        print(mess.format(argmin + 1, history.history['fbeta_score'][argmin]))
        
def is_game_over_move(move):
    return move in ('0-1', '1-0', '1/2-1/2')

#WARNING: don't use this anymore as RNN is much better it seems
def load_data(file, 
              embeddings,
              move_to_id_dict,
              split=(0.7, 0.15, 0.15), 
              embeddings_dir='../data/embeddings/5', 
              dimensions=50):

    def test_encoding(X, labels):
        X = np.asarray(X)
        y = np.asarray(labels)
        for i, label in enumerate(labels):
            if i < len(labels) - 1:
                if label in move_to_id_dict:
                    vec = embeddings[move_to_id_dict[label]]
                    assert (vec == X[i + 1][i]).all()
                else:
                    print('unknown move {}'.format(labels[i]))
    
    def encode_game(moves):
        
        encoded_X = []
        labels = []
        state = np.zeros((200, dimensions))

        # First move will have no board, so give it
        # a zeros matrix
        # labels.append(moves.pop(0))
        # encoded_X.append(np.copy(state))

        if len(moves) > 199:
            moves = moves[0:199]
            print(len(moves))
        
        for i, move in enumerate(moves):
            if move in move_to_id_dict:
                vec = embeddings[move_to_id_dict[move]]
            else:
                # empty if unknown
                vec = np.full(dimensions, 0.0)
            state[i] = vec
        
        # encoded_X.append(state)
        s = np.copy(state)
        
        while len(moves) > 0:
            labels.append(moves[0])
            s[len(moves) - 1] = np.zeros(dimensions)
            encoded_X.append(s)
            s = np.copy(s)
            moves.pop(0)
            
        encoded_X.reverse()

        # pdb.set_trace()
        # labels and encoded_X are now symetric,
        # (i.e. each encoded_X is an encoding of
        # the state of the board after the label),
        # but what we really need is for encoded_X
        # to describe the state of the board before
        # the label. So we insert an empty board 
        # state as the first element, and pop off
        # the last.
#         encoded_X.insert(0, np.zeros((200, d)))
        # encoded_X.pop(0)
        
        return encoded_X, labels
    
    def encode_moves(moves):
        
        encoded = []
        labels_array = []
        
        # find indices of all last moves in a game
        last_moves = [is_game_over_move(m) for m in moves]
        last_moves = [i for i, t in enumerate(last_moves) if t]
        
        start = 0
        for last_move in last_moves:
            game = moves[start:last_move + 1]
            # it is at least correct up to here...
            states, labels = encode_game(game)
            test_encoding(states, labels)
            [encoded.append(s) for s in states]
            [labels_array.append(l) for l in labels]
            start = last_move + 1

        return encoded, labels_array
    
    # split: train, val/dev, test
    with open(file, 'r') as f:
        moves = f.read().split()
        
    train = moves[0:int(len(moves) * split[0])]
    val = moves[len(train) + 1:len(train) + int(len(moves) * split[1])]
    test = moves[len(train) + len(val) + 1:]

    X_train, y_train = encode_moves(train[0:10000])
    X_val, y_val     = encode_moves(val[0:1500])
    X_test, y_test   = encode_moves(test[0:1000])
    return np.asarray(X_train), y_train, np.asarray(X_val), y_val, np.asarray(X_test), y_test

def create_model(num_inputs, 
                 num_outputs, 
                 verbose=False, 
                 save_dir=None,
                 hyper=None,
                 load_from=None):
    
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
                'type': 'adagrad',
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
    
    optimizer = get_optimizer(hyper)
    
    if verbose:
        print('loss:    {}'.format(hyper['loss']))
        print('metrics: {}'.format(*hyper['metrics']))
    
    model.compile(loss=hyper['loss'], optimizer=optimizer, metrics=hyper['metrics'])

    if save_dir:
        with open(os.path.join(save_dir, 'model.json'), 'w') as f:
            f.write(model.to_json())

    return model, callbacks

def load_model_from_checkpoint(model_dir, compile=True):
    '''Loads the best performing model from checkpoint_dir'''
    with open(os.path.join(model_dir, 'model.json'), 'r') as f:
        model = model_from_json(f.read())

    with open(os.path.join(model_dir, 'hyperparameters.json'), 'r') as f:
        hyper = json.load(f)
        optimizer = get_optimizer(hyper)

    newest_checkpoint = max(glob.iglob(model_dir + '/checkpoints/*.hdf5'), 
                            key=os.path.getctime)

    if newest_checkpoint:
        model.load_weights(newest_checkpoint)

    # If compile flag is true, use the compilation settings from the saved 
    # hyperparameters
    if compile:
        model.compile(loss=hyper['loss'], optimizer=optimizer, metrics=hyper['metrics'])

    return model


def get_optimizer(hyper):
    optim = hyper['optimizer']['type']
    if optim == 'rmsprop':
        optimizer = RMSprop(lr=hyper['optimizer']['lr'])
    elif optim == 'sgd':
        optimizer = SGD(lr=hyper['optimizer']['lr'])
    elif optim == 'adam':
        optimizer = Adam(lr=hyper['optimizer']['lr'])
    elif optim == 'adagrad':
        optimizer = Adagrad()
    else:
        raise Exception('Unsupported optimization algorithm')
    return optimizer

def remove_all_but_newest_checkpoint(model_dir):
    checkpoints = glob.glob(model_dir + '/checkpoints/*.hdf5')
    newest_checkpoint = max(checkpoints, key=os.path.getctime)
    checkpoints.remove(newest_checkpoint)
    for checkpoint in checkpoints:
        os.remove(checkpoint)
        
## This is the older way I was parsing. Newer method in
## load_data()
# def move_to_glove(move, embeddings, word_to_id):
#     if move in word_to_id:
#         return embeddings[word_to_id[move]]
#     else :
#         # return the "I've never seen that" vector
#         return np.full(len(embeddings[0]), 1, dtype=np.float32) 
#
# def encode_moves(moves, length):   
#     encoded = [move_to_glove(m) for m in moves]
#     while len(encoded) < length:
#         encoded.append(np.zeros(length))
#     return encoded
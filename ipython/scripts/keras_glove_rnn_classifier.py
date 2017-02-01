import sys, pdb, os, glob
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Add the parent directory to PYTHONPATH so that we can use utils.py
sys.path.append('../..')
import utils

def get_data_batch(moves, num_classes, window_size, batch_size=32):

    # copy 
    moves = list(moves)
    step = 1

    index = 0
    while True:
        
        input_moves = []
        output_moves = []

        if index + batch_size + window_size >= len(moves):
            index = 0

        for i in range(index, index + batch_size, step):
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

        # print('Moves not found in vector embedding dictionary:')
        # print(*unknown_moves)

        yield X, y
        index = index + batch_size

def eat_tail(model, start_moves, num_moves, num_classes, window_size):
    '''
    Predict moves, beginning with start_moves, where the previous
    predictions made by the model are included in the sequence
    fed to the model to create the next prediction.
    
    start_moves is a list of ASCII moves to use as the initial input
    sequence. start_moves must be of length window_length.

    Returns a list of ASCII moves of length num_moves
    '''

    sequence = start_moves
    batch_gen = get_data_batch(sequence, num_classes, window_size, batch_size=1)
    return_sequence = []
    for i in range(0, num_moves):
        X, y = next(batch_gen)
        next_move = id_to_move[np.argmax(model.predict(X))]
        sequence.pop(0)
        sequence.append(next_move)
        return_sequence.append(next_move)
        batch_gen = get_data_batch(sequence, num_classes, window_size, batch_size=1)
        # pdb.set_trace()
    return return_sequence

def load_model_from_checkpoint(model_dir):

    '''Loads the best performing model from checkpoint_dir'''
    with open(os.path.join(model_dir, 'model.json'), 'r') as f:
        model = model_from_json(f.read())

    epoch = 0

    newest_checkpoint = max(glob.iglob(model_dir + '/checkpoints/*.hdf5'), 
                            key=os.path.getctime)

    if newest_checkpoint: 
       epoch = int(newest_checkpoint[-22:-19])
       model.load_weights(newest_checkpoint)

    return model, epoch

def save_model(model, model_dir):
    with open(os.path.join(model_dir, 'model.json'), 'w') as f:
        f.write(model.to_json())

def get_model(window_size, num_classes, vector_dimensions, model_dir=None):

    epoch = 0
    # create new model
    if not model_dir:
        model = Sequential()
        model.add(LSTM(4096, return_sequences=False, 
                             batch_input_shape=(None, window_size, vector_dimensions)))
        model.add(Dropout(0.25))
        # model.add(LSTM(512, return_sequences=False))
        # model.add(Dropout(0.25))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    # load model from checkpoint
    else:
        model, epoch = load_model_from_checkpoint(model_dir)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=RMSprop(lr=0.001, decay=0.0005), 
                  metrics=['accuracy', 'fbeta_score', 'precision', 'recall'])

    return model, epoch

def get_callbacks(model_dir):
    
    callbacks = []
    
    # save model checkpoints
    filepath = os.path.join(model_dir, 
                            'checkpoints', 
                            'checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')
    callbacks.append(ModelCheckpoint(filepath, 
                                     monitor='val_acc', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='max'))

    # callbacks.append(EarlyStopping(monitor='val_loss', 
    #                                min_delta=0, 
    #                                patience=0, 
    #                                verbose=0, 
    #                                mode='auto'))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.5, 
                                       patience=3, 
                                       verbose=1, 
                                       mode='auto', 
                                       epsilon=0.0001, 
                                       cooldown=0, 
                                       min_lr=0))

    callbacks.append(TensorBoard(log_dir='./tensorboard-logs', 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False))

    return callbacks

def main():
    
    # Load move Dataset
    with open('../../data/test_moves.txt', 'r') as f:
        moves = f.read().split()

    # Only use a subset of moves, for now
    train_moves = moves[:3000000]
    # train_moves = moves
    n = len(train_moves)
    test_moves = moves[n:n + int(n * 0.15)]

    uniq_moves = list(set(moves))
    print('{} moves loaded'.format(len(moves)))
    print('{} unique moves in vector encoding'.format(len(labels)))
    print('{} unique moves in training set'.format(len(uniq_moves)))

    window_size = 10
    model_num = 34
    model_dir = 'data/models/long_running/{}'.format(model_num)

    # if the model dir doesn't exist
    # create it and checkpoints/
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        os.mkdir(os.path.join(model_dir, 'checkpoints'))

    # # create/load model
    model, epoch = get_model(window_size,  
                             num_classes=len(labels), 
                             vector_dimensions=d,
                             model_dir='data/models/long_running/{}'.format(28))

    print(model.summary())

    # saves json representation of model only,
    # to save the weights use a checkpoint callback
    save_model(model, model_dir)

    callbacks = get_callbacks(model_dir)

    # train data in batches so as not to have to load everything at once
    batch_train = get_data_batch(train_moves, len(labels), window_size, batch_size=256)
    batch_test  = get_data_batch(test_moves, len(labels), window_size, batch_size=256)

    history = model.fit_generator(batch_train, 
                                  validation_data=batch_test,
                                  nb_epoch=30,
                                  # samples_per_epoch=len(train_moves)/10,
                                  # nb_val_samples=len(test_moves)/10,
                                  samples_per_epoch=100000,
                                  nb_val_samples=len(test_moves),
                                  nb_worker=4,
                                  verbose=1,
                                  callbacks=callbacks)
   #                              initial_epoch=epoch,


    #clean up lesser checkpoint files
    utils.plot_model_results(history, save_dir=model_dir)  
    utils.remove_all_but_newest_checkpoint(model_dir)

    # predicted = model.predict_generator(batch_test, 256)
    # ids = [np.argmax(p) for p in predicted]
    # moves = [id_to_move[i] for i in ids]
    # for i in range(0, len(moves), 2):
    #     if i < len(moves) - 1:
    #         print('{}. {} {}'.format(i+1, moves[i], moves[i + 1]))    


    # length of moves sent as second parameter must be at least batch_size +
    # window_size
    # begin_game = 'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7'.split()
    # # end_game = 'Kf2 Bf5 Ra7 g6 Ra6+ Kc5 Ke1 Nf4 g3 Nxh3 Kd2 Kb5 Rd6 Kc5 Ra6 Nf2 g4 Bd3 Re6 1/2-1/2'.split()
    # moves = eat_tail(model, begin_game, 20, len(labels), window_size)
    # move_count = 1
    # for i in range(0, len(moves), 2):
    #     if i < len(moves) - 1:
    #         print('{}. {} {}'.format(move_count, moves[i], moves[i + 1]))   
    #         move_count = move_count + 1

    ## The generator can be itered over with...
    # for data in batch_load:
    #     X, y = data

if __name__ == '__main__':

    # some globals, I know... gross
    # Load Vector Embeddings
    d = 200 #dimensionality of word vectors
    data = utils.build_word_vector_matrix('../../data/embeddings/5/vectors_d{}.txt'.format(d))
    embeddings, labels, id_to_move, move_to_id = data

    main()
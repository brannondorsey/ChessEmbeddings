# this is a utility script and should not be run
# on its own w/out editing.
# Rather comment/uncomment code and run
# bits and pieces as needed per the task at hand.

"""
Run this only after having run load_parse_and_pickle_games.py
"""

import pickle
from itertools import chain


# takes a 2d array of moves from each game
# and flattens to 1d
def flatten_moves(moves_2d):
    return list(chain.from_iterable(moves_2d))


def pickle_all_unique_tokens(games, outfile):
    moves = [g.moves for g in games]
    flattened = flatten_moves(moves)
    print("{} moves".format(len(flattened)))
    flattened = list(set(flattened))
    print("{} unique moves (tokens)".format(len(flattened)))
    pickle.dump(flattened, open(outfile, "wb"))
    return flattened


# returns train, test split
def load_and_split(infile, split=0.90):
    with open(infile, "r") as f:
        moves = f.read()
        s = int(len(moves) * split)
        return moves[0:s], moves[s:]


def save_train_test(train, test):
    with open("data/train_moves.txt", "w") as f:
        f.write("".join(train))
    with open("data/test_moves.txt", "w") as f:
        f.write("".join(test))


def print_game(game):
    print("annotator", game.annotator)
    print("black", game.black)
    print("date", game.date)
    print("eco", game.eco)
    print("event", game.event)
    print("eventdate", game.eventdate)
    print("fen", game.fen)
    print("mode", game.mode)
    print("moves", game.moves)
    print("plycount", game.plycount)
    print("result", game.result)
    print("round", game.round)
    print("site", game.site)
    print("termination", game.termination)
    print("time", game.time)
    print("timecontrol", game.timecontrol)
    print("white", game.white)
    print("whiteelo", game.whiteelo)


## parse and save unique tokens
# tokens = pickle_all_unique_tokens(games, 'data/tokens.pickle')

## load games
with open("data/datasets/all.pickle", "rb") as f:
    games = pickle.load(f)
print("loaded {} games".format(len(games)))

with open("data/moves_from_{}_games.txt".format(len(games)), "w") as f:
    for i, game in enumerate(games):
        if not game:
            print("No game at index {}".format(i))
            continue
        if i % 1000 == 0:
            print("writing game {} of {}".format(i, len(games)))
        try:
            game.moves.remove("1-0")
        except ValueError:
            pass
        try:
            game.moves.remove("0-1")
        except ValueError:
            pass
        try:
            game.moves.remove("1/2-1/2")
        except ValueError:
            pass
        # Add "start", "end", and chop of the result
        moves = ["<start>"] + game.moves[0 : len(game.moves)] + ["<end>"]
        text = " ".join(moves) + " "
        f.write(text)
    print("wrote all games to file")

# train, test = load_and_split("data/14444236_moves.txt", 0.90)
# print("train: {}, test: {}".format(len(train), len(test)))

# save_train_test(train, test)

# load unique tokens
# with open('data/tokens.pickle', 'rb') as f:
# 	tokens = pickle.load(f)
# print('loaded {} tokens'.format(len(tokens)))

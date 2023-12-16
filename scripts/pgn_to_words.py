# this is a utility script and should not be run
# on its own w/out editing.
# Rather comment/uncomment code and run
# bits and pieces as needed per the task at hand.

"""
Run this only after having run load_parse_and_pickle_games.py
"""

import pickle


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

## load games
# with open("data/datasets/all.pickle", "rb") as f:
#     games = pickle.load(f)
# print("loaded {} games".format(len(games)))

# with open("data/moves_from_{}_games.txt".format(len(games)), "w") as f:
#     for i, game in enumerate(games):
#         if not game:
#             print("No game at index {}".format(i))
#             continue
#         if i % 1000 == 0:
#             print("writing game {} of {}".format(i, len(games)))
#         try:
#             game.moves.remove("1-0")
#         except ValueError:
#             pass
#         try:
#             game.moves.remove("0-1")
#         except ValueError:
#             pass
#         try:
#             game.moves.remove("1/2-1/2")
#         except ValueError:
#             pass
#         # Add "start", "end", and chop of the result
#         moves = ["<start>"] + game.moves[0 : len(game.moves)] + ["<end>"]
#         text = " ".join(moves) + " "
#         f.write(text)
#     print("wrote all games to file")

train, test = load_and_split("data/3561458_moves.txt", 0.90)
print("train: {}, test: {}".format(len(train), len(test)))

save_train_test(train, test)

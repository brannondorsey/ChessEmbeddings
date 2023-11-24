# this is a utility script and should not be run
# on its own w/out editing.
# Rather comment/uncomment code and run
# bits and pieces as needed per the task at hand.
import os
import pgn
import codecs, pickle
from typing import List
from itertools import chain
from concurrent.futures import ProcessPoolExecutor


# takes a 2d array of moves from each game
# and flattens to 1d
def flatten_moves(moves_2d):
    return list(chain.from_iterable(moves_2d))


def split_pgn_contents_into_batches(file_contents: str, batches: int):
    result_markers = ["\n1-0\n", "\n0-1\n", "\n1/2-1/2\n"]
    game_starts = [0]  # List of indices where games start

    # Find all game start positions
    for marker in result_markers:
        start = 0
        while start < len(file_contents):
            start = file_contents.find(marker, start)
            if start == -1:
                break
            # Move to the end of this game result marker
            start += len(marker)
            if start < len(file_contents):
                game_starts.append(start)

    # Remove duplicate indices and sort
    game_starts = sorted(set(game_starts))

    # Split into batches
    batches_list = []
    batch_size = len(game_starts) // batches
    for i in range(batches):
        start_index = game_starts[i * batch_size]
        end_index = game_starts[min((i + 1) * batch_size, len(game_starts)) - 1]
        batches_list.append(file_contents[start_index:end_index])

    return batches_list


def parse_pgn_batch(pgn_batch):
    print(f"parsing batch of {len(pgn_batch)} characters")
    try:
        games = pgn.loads(pgn_batch)
    except AttributeError:
        print("AttributeError")
        return []
    print(f"parsed {len(games)} games")
    return games


def load_parse_and_pickle_games(infile, outfile, num_workers=os.cpu_count() or 1):
    games = None
    with codecs.open(infile, encoding="ascii", errors="replace") as f:
        pgn_batches = split_pgn_contents_into_batches(f.read(), num_workers)
        for i, batch in enumerate(pgn_batches):
            with open(f"/tmp/batch_{i}.pgn", "w") as f:
                f.write(batch)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(parse_pgn_batch, pgn_batches)
            games = list(chain.from_iterable(results))
            print("Total games parsed:", len(games))
    pickle.dump(games, open(outfile, "wb"))
    return games


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
# with open('data/10_million_lines.pickle', 'rb') as f:
# 	games = pickle.load(f)
# print('loaded {} games'.format(len(games)))

games = load_parse_and_pickle_games(
    "data/100_million_lines.pgn", "data/100_million_lines.pickle"
)

# # ## parse and save all moves
# moves = flatten_moves([g.moves for g in games])
# print('found {} moves'.format(len(moves)))
# text = ' '.join(moves)
# with open('data/{}_moves.txt'.format(len(moves)), 'w') as f:
# 	f.write(text)
# 	print('wrote {} chars to file'.format(len(text)))

# train, test = load_and_split("data/14444236_moves.txt", 0.90)
# print("train: {}, test: {}".format(len(train), len(test)))

# save_train_test(train, test)

# load unique tokens
# with open('data/tokens.pickle', 'rb') as f:
# 	tokens = pickle.load(f)
# print('loaded {} tokens'.format(len(tokens)))

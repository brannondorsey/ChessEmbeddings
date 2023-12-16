"""
Generate data/datasets/all.csv using data/datasets/all.pickle

Run this only after having run load_parse_and_pickle_games.py
"""

import pickle
import csv


def print_game(game):
    obj = game_to_dict(game)
    for key, value in obj.items():
        print(f"{key}: {value}")


def optional(game, attr: str) -> str | None:
    return getattr(game, attr) if hasattr(game, attr) else None


def game_to_dict(game):
    return {
        "annotator": game.annotator,
        "black": game.black,
        "date": game.date,
        "eco": optional(game, "eco"),
        "event": game.event,
        "eventdate": optional(game, "eventdate"),
        "fen": game.fen,
        "mode": game.mode,
        "moves": ",".join(game.moves),
        "plycount": game.plycount,
        "result": game.result,
        "round": game.round,
        "site": game.site,
        "termination": game.termination,
        "time": game.time,
        "timecontrol": game.timecontrol,
        "white": game.white,
        "whiteelo": optional(game, "whiteelo"),
    }


# load games
with open("data/datasets/all.pickle", "rb") as f:
    games = pickle.load(f)
print("loaded {} games".format(len(games)))

with open("data/datasets/all.csv", "w") as f:
    columns = game_to_dict(games[0]).keys()
    writer = csv.DictWriter(
        f,
        columns,
    )
    writer.writeheader()
    for i, game in enumerate(games):
        if i % 1000 == 0 and i != 0:
            print(i)
        obj = game_to_dict(game)
        writer.writerow(obj)

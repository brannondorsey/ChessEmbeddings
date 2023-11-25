"""
Used to load a PGN file, parse the games, and pickle them to a file
for faster loading in the future without the overhead of parsing.

Run like:
    PYTHONPATH=. python3 scripts/load_parse_and_pickle_games.py

WARNING: This script uses A LOT of memory. I ran this on all.pgn 
(2.9GB of text, 3,561,458 games) and it took all 32GB of my RAM and ~10GB of swap.
That's OK, because I had the hardware and that's the most data I
needed to parse, but this script could be updated to lazy load and
save data in the future if necessary.
"""
import os
import pgn
import codecs, pickle
from src.TaskManager import TaskManager

INPUT_FILENAME = "data/datasets/all.pgn"
OUTPUT_FILENAME = f"{INPUT_FILENAME[:-4]}.pickle"


def split_pgn_games(file_contents: str):
    result_markers = ["\n1-0\n", "\n0-1\n", "\n1/2-1/2\n"]
    start_index = 0
    nearest_marker = ""

    # Iterate over the file contents to find each game
    while start_index < len(file_contents):
        # Find the nearest game result marker
        nearest_marker_index = len(file_contents)
        for marker in result_markers:
            marker_index = file_contents.find(marker, start_index)
            if 0 <= marker_index < nearest_marker_index:
                nearest_marker_index = marker_index
                nearest_marker = marker

        # If a game end is found, yield the game
        if nearest_marker_index != len(file_contents):
            # Include the marker in the game text
            end_index = nearest_marker_index + len(nearest_marker)
            yield file_contents[start_index:end_index]
            start_index = (
                end_index + 1
            )  # Move past the newline character after the marker
        else:
            # If no more games are found, yield the last game and break
            if start_index < len(file_contents):
                yield file_contents[start_index:]
            break


def produce_new_task():
    with codecs.open(INPUT_FILENAME, encoding="ascii", errors="replace") as f:
        for i, pgn_string in enumerate(split_pgn_games(f.read())):
            yield i, pgn_string


def consumer_func(item):
    i, pgn_string = item
    try:
        game = pgn.loads(pgn_string)[0]
    except IndexError:
        return i, None
    return i, game


unflushed_games = []


def process_finished_task(result, all_tasks_finished: bool):
    global unflushed_games
    if len(unflushed_games) % 1000 == 0:
        print(f"Games parsed so far: {len(unflushed_games)}")
    if all_tasks_finished:
        unflushed_games.sort(key=lambda x: x[0])
        with open(OUTPUT_FILENAME.format(0), "wb") as f:
            pickle.dump([x[1] for x in unflushed_games], f)
        print(f"Flushed {len(unflushed_games)} games to {OUTPUT_FILENAME}")
    else:
        unflushed_games.append(result)


if __name__ == "__main__":
    task_manager = TaskManager(
        num_consumers=os.cpu_count() or 1,
        produce_new_task=produce_new_task(),
        process_finished_task=process_finished_task,
        consumer_func=consumer_func,
    )
    task_manager.run()

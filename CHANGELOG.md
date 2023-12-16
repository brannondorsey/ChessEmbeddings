# Changelog

## 2.0

* Train all embeddings for 1,000 iterations, instead of 200 used in 1.0.
* Replace use of `1-0`, `0-1`, and `1/2-1/2` as delimiters of games in GloVe training set with `<start>` and `<end>` tokens. The result of each game is no longer inherently preserved or learned during the embedding process.
* Added power-of-two versions of pre-trained embeddings in addition to the sizes used in the original GloVe paper. You will now find embeddings for 8, 16, 32, 64, 128, and 512 embeddings in [`data/embeddings/`](data/embeddings) in addition to the previous dimensions (10, 25, 50, 100, 200, 300).
* Train embeddings on full 3.5 million game corpus instead of the 100,000+ game sample used in 1.0. That's 2.9 GB of data instead of 40 MB.
* Created an optimized version of `load_parse_and_pickle()` which uses `multiprocessing` and a new[`TaskManager`](src/TaskManager.py) class that implements an efficient consumer/producer concurrency model. See [`scripts/load_parse_and_pickle.py`](scripts/load_parse_and_pickle_games.py) for details.
* Add `requirements.txt` for reproducibility
* Update GloVe submodule to latest `master`
* Add `scripts/pgn_to_csv.py` to convert `all.pgn` to a CSV for easier parsing/manipulation in downstream tasks.

## 1.0

* Original 2017 release. See this tree for a [snapshot](https://github.com/brannondorsey/ChessEmbeddings/tree/e521638b39ea4af1efa4c62ab519406324fea385) of the original 1.0 release.

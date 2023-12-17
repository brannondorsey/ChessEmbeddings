# Chess Embeddings

Pre-trained [GloVe](git@github.com:stanfordnlp/GloVe.git) word embeddings created from ~3.5 million chess games from the [ChessDB dataset](https://chessdb.sourceforge.net/).

Embeddings were trained on ~3GB of convenient PGN-formatted data made available via [this dataset](http://chess-research-project.readthedocs.io/en/latest/) (mirror download also available [here](https://brannondorsey.s3.amazonaws.com/ChessEmbeddings/all.pgn.zip)). You can find these embeddings inside `data/embeddings/`. I've also provided scripts and instructions to easily create your own embeddings using more data from the above dataset if you prefer.

## Downloads

* [Pre-trained Chess embeddings](https://brannondorsey.s3.amazonaws.com/ChessEmbeddings/embeddings.zip) (215 MB)
* [Text file used to train these embeddings](https://brannondorsey.s3.amazonaws.com/ChessEmbeddings/moves_from_3561458_games.zip) (374 MB)
* [Original ChessDB PGN dataset](https://brannondorsey.s3.amazonaws.com/ChessEmbeddings/all.pgn.zip) (830 MB)
* [Original ChessDB PGN dataset transformed to CSV format](https://brannondorsey.s3.amazonaws.com/ChessEmbeddings/all.csv.zip) (540 MB)
* [Python pickle of all into a `pgnparser.PGNGame` array](https://brannondorsey.s3.amazonaws.com/ChessEmbeddings/all.pickle.zip) (requires ~32GB of RAM to load) (656 MB)

## Creating Your Own Embeddings

Start by cloning and building [Stanford's GloVe repo](https://github.com/stanfordnlp/GloVe). I've provided a script that does this for you with:

```bash
./setup.sh
```

The GloVe tools require a file filled with words (in our case chess moves) seperated by spaces to train the embedding vectors with. Because GloVe trains word vectors that represent word colocation statistics, the order of the words in this input file is very important. 

I've included the file that I used to generate the embeddings included with this repo (created from the above chess dataset) in `data/train_moves.txt`. This file contains the first 10,108,811 moves in the chess dataset. You will also find a `data/test_moves.txt` with 4,335,426 moves that the word vectors have not been trained with. In both of these files, games are not explicitly seperated, however doing so is as easy as splitting on moves `0-1`, `1-0`, or `1/2-1/2`. 

If you would like to create embeddings using more moves from the dataset (and there are certaintly a **lot** more moves), you must create a file similar to `data/train_moves.txt`. I intend to provide a script for doing this as well, but haven't gotten around to it yet (see `scripts` for a messy file with bits and pieces that might help you. Here be dragons.)

Once this is done, open `create_embeddings.sh` and edit the value of the `WORD_FILE` variable to point towards your word file.

There are many other variables here that you can change to customize the training of your GloVe vectors. Once you've edited the appropriate variables, create the embeddings with:

```bash
./create_embeddings.sh
```

Your new embeddings, by default, will override the existing files in `data/embeddings`

## Analyzing the Data

I've provided a few Jupyter notebooks that analyze/visualize the moves in `data/train_moves.txt` in different ways. Poke around `notebooks` to learn some interesting things about chess :) 

# Chess Embeddings

Pre-trained [GloVe](git@github.com:stanfordnlp/GloVe.git) word embeddings of ~40MB chess moves. 

Current embeddings were trained on only a small subset of the ~3GB of data from [this dataset](http://chess-research-project.readthedocs.io/en/latest/). You can find these embeddings inside `data/embeddings/`. I've also provided scripts and instructions to easily create your own embeddings using more data from the above dataset if you prefer.

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

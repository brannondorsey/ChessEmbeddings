#!/bin/bash

# Last run time: ./create_embeddings.sh  373374.72s user 1456.40s system 1148% cpu 9:04:07.97 total
WORD_FILE=data/moves_from_3561458_games.txt
OUT_DIR=data/embeddings

MIN_COUNT=5
WINDOW_SIZE=15
# Includes the original vector sizes from the GloVe paper + their power of two equivalents
DIMENSIONS=( 8 10 16 25 32 50 64 100 128 200 256 300 512 )
ITERATIONS=1000
NUM_THREADS=12

mkdir -p $OUT_DIR/

GloVe/build/vocab_count -min-count $MIN_COUNT -verbose 2 < $WORD_FILE > "$OUT_DIR/vocab.txt"
GloVe/build/cooccur -memory 4.0 -vocab-file "$OUT_DIR/vocab.txt" -verbose 2 -window-size $WINDOW_SIZE < $WORD_FILE > "$OUT_DIR/cooccurrence.bin"
GloVe/build/shuffle -memory 4.0 -verbose 2 < "$OUT_DIR/cooccurrence.bin" > "$OUT_DIR/cooccurrence.shuf.bin"

for D in "${DIMENSIONS[@]}"
do
	GloVe/build/glove -save-file "$OUT_DIR/vectors_d$D" -threads $NUM_THREADS \
					  -input-file "$OUT_DIR/cooccurrence.shuf.bin" \
				      -x-max 10 -iter $ITERATIONS -vector-size $D \
				      -binary 2 -vocab-file "$OUT_DIR/vocab.txt" -verbose 2
done

rm $OUT_DIR/cooccurrence.bin
rm $OUT_DIR/cooccurrence.shuf.bin

#save these settings to file
echo "WORD_FILE=$WORD_FILE"                      >  $OUT_DIR/settings.txt
echo "OUT_DIR=$OUT_DIR"                          >> $OUT_DIR/settings.txt
echo "MIN_COUNT=$MIN_COUNT"                      >> $OUT_DIR/settings.txt
echo "WINDOW_SIZE=$WINDOW_SIZE"                  >> $OUT_DIR/settings.txt
echo "DIMENSIONS=$(echo ${DIMENSIONS[*]// /|})"  >> $OUT_DIR/settings.txt
echo "ITERATIONS=$ITERATIONS"                    >> $OUT_DIR/settings.txt


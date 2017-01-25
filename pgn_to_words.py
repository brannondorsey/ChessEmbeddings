# this is a utility script and should not be run
# on its own w/out editing. 
# Rather comment/uncomment code and run
# bits and pieces as needed per the task at hand.

import pgn
import codecs, pickle

# takes a 2d array of moves from each game
# and flattens to 1d
def flatten_moves(moves_2d):
	flattened = []
	while (len(moves_2d) > 0):
		flattened = flattened + moves_2d.pop(0)
	return flattened

def load_parse_and_pickle_games(infile, outfile):
	games = None
	with codecs.open(infile, encoding='ascii', errors='replace') as f:
		contents = f.read()
		games = pgn.loads(contents)

	pickle.dump(games, open(outfile, 'wb'))
	return games

def pickel_all_unique_tokens(games, outfile):
	moves = [g.moves for g in games]
	flattened = flatten_moves(moves)
	print('{} moves'.format(len(flattened)))
	flattened = list(set(flattened))
	print('{} unique moves (tokens)'.format(len(flattened)))
	pickle.dump(flattened, open(outfile, 'wb'))
	return flattened

def print_game(game):
	print('annotator', game.annotator)
	print('black', game.black) 
	print('date', game.date)
	print('eco', game.eco)
	print('event', game.event)
	print('eventdate', game.eventdate)
	print('fen', game.fen)
	print('mode', game.mode)
	print('moves', game.moves)
	print('plycount', game.plycount)
	print('result', game.result)
	print('round', game.round)
	print('site', game.site)
	print('termination', game.termination)
	print('time', game.time)
	print('timecontrol', game.timecontrol)
	print('white', game.white)
	print('whiteelo', game.whiteelo)

## parse and save unique tokens
# tokens = pickel_all_unique_tokens(games, 'data/tokens.pickle')

# load games
# with open('data/10_million_lines.pickle', 'rb') as f:
# 	games = pickle.load(f)
# print('loaded {} games'.format(len(games)))

games = load_parse_and_pickle_games('data/10_million_lines.pgn', 'data/10_million_lines.pickle')

## parse and save all moves
moves = flatten_moves([g.moves for g in games])
print('found {} moves'.format(len(moves)))
text = ' '.join(moves)
with open('data/moves.txt', 'w') as f:
	f.write(text)
	print('wrote {} chars to file'.format(len(text)))

# load unique tokens
# with open('data/tokens.pickle', 'rb') as f:
# 	tokens = pickle.load(f)
# print('loaded {} tokens'.format(len(tokens)))

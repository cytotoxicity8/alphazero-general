import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.hex.hex import Game, BOARD_SIZE, OBSERVATION_BOARD_SIZE, NUM_CHANNELS, MAX_TURNS, CANONICAL_STATE
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.envs.hex.players import GTPPlayer
from alphazero.utils import dotdict
import cProfile

BASE_SIZE = OBSERVATION_BOARD_SIZE * OBSERVATION_BOARD_SIZE * NUM_CHANNELS

args = get_args(dotdict({
    'run_name': 'hex_{0}x{0}_observing_{1}x{1}x{2}_Canonical_NoSwap'.format(BOARD_SIZE, OBSERVATION_BOARD_SIZE, NUM_CHANNELS),
    'workers': mp.cpu_count(),
    'startIter': 1,
    'numIters': 100,
    'numWarmupIters': 2,
    "num_stacked_observations" : 1,
    'process_batch_size': 256,
    'train_batch_size': 256,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * mp.cpu_count(),
    '_num_players' : 2,
    'symmetricSamples': True,
    'train_on_past_data' : False,
    'past_data_run_name' : "human4Layers",
    'past_data_chunk_size': 2,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'mctsCanonicalStates': CANONICAL_STATE, 
    'deterministic': False,
    'simplifiedResultQueue':False,
    'numMCTSSims': 200, 
    'numFastSims': 20,
    'probFastSim': 0.75,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 256,
    'arenaCompare': 128,
    'arena_batch_size': 32,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'baselineCompareFreq': 0.5,
    'compareWithPast': True,
    'pastCompareFreq': 0.5,
    'cpuct': 5,
    'fpu_reduction': 0,
    'load_model': True,
}),
    model_gating=True,
    max_gating_iters=None,
    max_moves=MAX_TURNS,

    lr=0.01,
    num_channels=128,
    depth=12,
    value_head_channels=32,
    policy_head_channels=32,
)
args.scheduler_args.milestones = [75, 150]

def doTrain():
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()


if __name__ == "__main__":
    #cProfile.runctx("doTrain()", globals(), locals(), "trainhex_5x5_observing_5x5x4_basicSym.prof")
    doTrain()

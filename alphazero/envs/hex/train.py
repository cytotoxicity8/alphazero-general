import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach, get_args,ModeOfGameGen
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.hex.hex import Game, BOARD_SIZE, OBSERVATION_BOARD_SIZE, NUM_CHANNELS, MAX_TURNS, CANONICAL_STATE
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.envs.hex.players import GTPPlayer
from alphazero.utils import dotdict
import cProfile

BASE_SIZE = OBSERVATION_BOARD_SIZE * OBSERVATION_BOARD_SIZE * NUM_CHANNELS

def hexxyAgs(i):
    return dotdict({'fpu_reduction':-0.151 + 0.05*i,
                    'root_noise_frac': 0.01 + 0.03 * i%4,
                    'cpuct': 3+0.7*(i%3),
                    'lr' : 0.015,
                    'value_loss_weight' : 1.4})

args = get_args(dotdict({
    'run_name': 'hex_{0}x{0}_observing_{1}x{1}x{2}_NeuroHex_Medium_JustBridge'.format(BOARD_SIZE, OBSERVATION_BOARD_SIZE, NUM_CHANNELS),
    'workers': mp.cpu_count(),
    'startIter': 0,
    'numIters': 100,
    'numWarmupIters': 1,
    "num_stacked_observations" : 1,
    'process_batch_size': 128,
    'train_batch_size': 256,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * mp.cpu_count(),
    '_num_players' : 2,

    'withPopulation' : True,
    'modeOfAssigningWork' : ModeOfGameGen.ROUND_ROBIN_EVERY_TIME,
    'roundRobinAsSelfPlay' : False,
    'populationSize' : 16,
    'getInitialArgs' : hexxyAgs,
    'roundRobinFreq' : 1,
    'roundRobinGames' : 3,
    'percentageKilled' : 0.2,

    'symmetricSamples': True,
    'train_on_past_data' : False,
    'past_data_run_name' : "human4Layers",
    'past_data_chunk_size': 2,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'mctsCanonicalStates': CANONICAL_STATE,
    'numMCTSSims': 200, 
    'numFastSims': 20,
    'probFastSim': 0.75,
    'compareWithBaseline': True,
    'arenaCompare': 128,
    'arena_batch_size': 64,
    'arenaTemp': 0.5,
    'arenaMCTS': True,
    'baselineCompareFreq': 4,
    'compareWithPast': True,
    'model_gating' : True,
    'pastCompareFreq': 4,
    'cpuct': 5,
    'fpu_reduction': 0,
    'load_model': True,
}),
    model_gating=True,
    max_gating_iters=None,
    max_moves=MAX_TURNS,

    lr=0.02,
    num_channels=32,
    depth=8,
    value_head_channels=16,
    policy_head_channels=16,
)
args.scheduler_args.milestones = [75, 150]

def doTrain():
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()


if __name__ == "__main__":
    #cProfile.runctx("doTrain()", globals(), locals(), "trainhex_5x5_observing_5x5x4_basicSym.prof")
    doTrain()

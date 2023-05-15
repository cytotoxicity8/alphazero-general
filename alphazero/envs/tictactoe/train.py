from torch import multiprocessing as mp 
import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args, ModeOfGameGen
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tictactoe.tictactoe import Game
from alphazero.utils import dotdict
import cProfile

def argsss(x):
    return dotdict({'cpuct': 0.5*(x//2), 'fpu_reduction': 0.5*(x%2)})

args = get_args(
    run_name='tictacgraph_5_noDropout_norm_deeper_DFS_batch',
    numIters=50,
    withPopulation=False,
    populationSize= 16,
    roundRobinFreq=1,
    percentageKilled=0.3,
    modeOfAssigningWork= ModeOfGameGen.ONE_PER_WORKER,
    getInitialArgs= argsss,
    workers=12,
    cpuct=2,
    numMCTSSims=250,
    probFastSim=0,
    numWarmupIters=1,
    baselineCompareFreq=1,
    pastCompareFreq=5,
    arenaBatchSize=16,
    arenaCompare=128,
    process_batch_size=128,
    train_batch_size=256,
    gamesPerIteration=20*256,
    lr=0.01,
    num_channels=16,
    middle_layers=[2*16],
    constant_edges=False,
    depth=3,
    value_head_channels=16,
    policy_head_channels=16,
    value_dense_layers=[],
    policy_dense_layers=[],
    compareWithBaseline=True,
    skipSelfPlayIters=None,
    compareWithPast=True,
    calculateElo=True,
    nnet_type='graphnet',
    symmetricSamples=False,
    eloMCTS=25,
    eloGames=10,
    eloMatches=10,
    #arenaBatched=False
)

def doTrain():
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()

if __name__ == "__main__":
    #mp.set_start_method('spawn', force=True)
    doTrain()
    #cProfile.runctx("doTrain()", globals(), locals(), "tictacgraph.prof")

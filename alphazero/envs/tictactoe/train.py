import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tictactoe.tictactoe import Game
from alphazero.utils import dotdict
def argsss(x):
    return dotdict({'cpuct': 0.5*(x//2), 'fpu_reduction': 0.5*(x%2)})

args = get_args(
    run_name='tictacfour',
    withPopulation=True,
    populationSize= 16,
    roundRobinFreq=5,
    percentageKilled=0.3,

    getInitialArgs= argsss,
    workers=12,
    cpuct=2,
    numMCTSSims=25,
    probFastSim=0.5,
    numWarmupIters=0,
    baselineCompareFreq=1,
    pastCompareFreq=5,
    arenaBatchSize=128,
    arenaCompare=128,
    process_batch_size=128,
    train_batch_size=512,
    gamesPerIteration=10*512,
    lr=0.01,
    num_channels=32,
    depth=4,
    value_head_channels=4,
    policy_head_channels=4,
    value_dense_layers=[128, 64],
    policy_dense_layers=[128],
    compareWithBaseline=True,
    compareWithPast=True,

    eloMCTS=25,
    eloGames=10,
    eloMatches=10,
    #arenaBatched=False
)

if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()

from torch import multiprocessing as mp 
import pyximport; pyximport.install()

import os
import sys
sys.path.append(os.path.abspath('/root/share/Real/KAIST/word_chain/alphazero-general'))
os.chdir('/root/share/Real/KAIST/word_chain/alphazero-general')


from alphazero.Coach import Coach, get_args, ModeOfGameGen
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.wordchain.wordchain import Game
from alphazero.utils import dotdict
import cProfile

def argsss(x):
    return dotdict({'cpuct': 0.5*(x//2), 'fpu_reduction': 0.5*(x%2)})

args = get_args(
    run_name='test_0717',
    numIters=50,
    withPopulation=False,
    populationSize= 16,
    roundRobinFreq=1,
    percentageKilled=0.3,
    modeOfAssigningWork= ModeOfGameGen.ONE_PER_WORKER,
    getInitialArgs= argsss,
    workers=12, #mp.cpu_count()
    cpuct=2,
    numMCTSSims=1000,
    probFastSim=0,
    numWarmupIters=1,
    baselineCompareFreq=2, #이거 2로
    pastCompareFreq=1,#이거 1로
    arenaBatchSize=16,
    arenaCompare=40,#이거 40으로
    process_batch_size=128, #원래 128
    train_batch_size=32,
    gamesPerIteration= 12*32,#mp.cpu_count() 고려
    lr=0.01,
    num_channels=128, #키워보기?
    middle_layers=[128], #키워보기?
    constant_edges=False,
    depth=3,
    value_head_channels=128, #키워보기?
    policy_head_channels=128, #둘이 같아야 하는 듯
    value_dense_layers=[256,128], #키워보기?
    policy_dense_layers=[256,128], #키워보기?
    compareWithBaseline=True,
    skipSelfPlayIters=None,
    compareWithPast=True,
    min_next_model_winrate= 0.6,#이거 0.6으로
    calculateElo=True,
    nnet_type='custom_graphmodel1',
    gnn_type = 'GAT',
    symmetricSamples=False,
    eloMCTS=100,
    eloGames=10,
    eloMatches=10,
    arenaBatched=False,

    use_head_embed = True
)

def doTrain():
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    doTrain()
    #cProfile.runctx("doTrain()", globals(), locals(), "tictacgraph.prof")

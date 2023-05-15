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

ARGDICT = [[0.30223265386029413, 0.27032637746704324, 7.355717008135116, 0.02297419387526726, 1.132524458834969] ,
[0.29171447345606727, 0.3163431352153565, 4.203009374436876, 0.01231174523781206, 1.9129675992419115] ,
[0.2423986196519827, 0.46136587659751765, 4.011967206436234, 0.03235940918385223, 1.2269304924322257] ,
[0.19851926997537256, 0.3090396145790576, 4.0882946123256, 0.013717969106878029, 1.3305659958571723] ,
[0.2539937731630696, 0.3421303459118472, 5.269783287598295, 0.02335340568726333, 1.1079234167667662] ,
[0.08929608156877261, 0.12271437748000119, 3.366769919618962, 0.01204805586249657, 1.7709828381065407] ,
[0.2833323020860575, 0.2894470763908246, 6.2286008872151495, 0.027188379884577164, 1.2046502900704963] ,
[0.26326829032885407, 0.33785675608713567, 3.9528086972646945, 0.012420050198388965, 1.6154473924084796] ,
[-0.0011444923487476083, 0.11604258364048475, 3.064497817143979, 0.014270696795002136, 1.2292171061829242] ,
[0.23638977861138202, 0.3107499820283833, 4.488536450117886, 0.019107222801481697, 1.508541842178417] ,
[0.25059378812467026, 0.35486872102825456, 4.064796581018562, 0.015263340143432459, 1.6425008931212028] ,
[0.20290964811890422, 0.39248403732658915, 4.529675528566823, 0.019970578164613104, 1.6254735962070708] ,
[0.3157530717378875, 0.4277062153763452, 3.684862896105816, 0.014674452673558927, 1.3384242644643933] ,
[0.20318558444570423, 0.4434310455390036, 3.671988590052009, 0.029595707115268217, 1.4895177650400664] ,
[0.2233220511198021, 0.38684714227173594, 4.555829940743138, 0.02581015002875224, 1.4993894876712606] ,
[0.2198539576623912, 0.4131511394714909, 3.9205201630551634, 0.018000479237951162, 1.8905121359654364]]


def hexxyAgs(i):
    return dotdict({'fpu_reduction':-0.151 + 0.05*i,
                    'root_noise_frac': 0.01 + 0.03 * i%4,
                    'cpuct': 3+0.7*(i%3),
                    'lr' : 0.015,
                    'value_loss_weight' : 1.4})

def hexxyAgs2(i):
    return dotdict({'fpu_reduction':ARGDICT[i][0],
                    'root_noise_frac': ARGDICT[i][1],
                    'cpuct': ARGDICT[i][2],
                    'lr' : ARGDICT[i][3],
                    'value_loss_weight' : ARGDICT[i][4]})

args = get_args(dotdict({
    'run_name': 'ahex_{0}x{0}_observing_{1}x{1}x{2}_NeuroHex_Virtual_Bridge_and_Ziguarat'.format(BOARD_SIZE, OBSERVATION_BOARD_SIZE, NUM_CHANNELS),
    'checkpoint': '/media/pooki/OxLecture/checkpoint',
    'workers': mp.cpu_count(),
    'startIter': 0,
    'numIters': 200,
    'numWarmupIters': 0,
    "num_stacked_observations" : 1,
    'process_batch_size': 128,
    'train_batch_size': 256,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * mp.cpu_count(),
    '_num_players' : 2,

    'withPopulation' : True,
    'modeOfAssigningWork' : ModeOfGameGen.ROUND_ROBIN_EVERY_TIME,
    'roundRobinAsSelfPlay' : False,
    'forceArgs':False,
    'populationSize' : 16,
    'getInitialArgs' : hexxyAgs,
    'roundRobinFreq' : 1,
    'roundRobinGames' : 3,
    'percentageKilled' : 0.2,

    'symmetricSamples': True,
    'train_on_past_data' : False,
    'past_data_run_name' : "human",
    'past_data_chunk_size': 2,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'mctsCanonicalStates': CANONICAL_STATE,
    'numMCTSSims': 200, 
    'numFastSims': 20,
    'probFastSim': 0.75,
    'compareWithBaseline': True,
    'arenaCompare': 128,
    'arena_batch_size': 16,
    'arenaTemp': 0.5,
    'arenaMCTS': True,
    'baselineCompareFreq': 4,
    'model_gating': False,
    'compareWithPast': False,
    'pastCompareFreq': 4,
    'cpuct': 5,
    'fpu_reduction': 0,
    'load_model': True,
    'max_gating_iters':None,
    'max_moves':MAX_TURNS,
    'lr':0.02,
    'num_channels':32,
    'depth':8,
    'value_head_channels':16,
    'policy_head_channels':16,
    })
)
"""
args = get_args(dotdict({
    'run_name': 'ahex_{0}_Graph_128HiddenAfterHuman'.format(NUM_CHANNELS),
    'workers': mp.cpu_count(),
    'startIter': 0,
    'numIters': 50,
    'numWarmupIters': 0,
    "num_stacked_observations" : 1,
    'process_batch_size': 128,
    'train_batch_size': 256,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * mp.cpu_count(),
    '_num_players' : 2,

    'withPopulation' : False,
    'modeOfAssigningWork' : ModeOfGameGen.ONE_PER_WORKER,
    'roundRobinAsSelfPlay' : False,
    'populationSize' : 16,
    'roundRobinFreq' : 1,
    'roundRobinGames' : 3,
    'percentageKilled' : 0.2,

    'symmetricSamples': True,
    'train_on_past_data' : False,
    'past_data_run_name' : "human",
    'past_data_chunk_size': 2,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'mctsCanonicalStates': CANONICAL_STATE,
    'numMCTSSims': 200, 
    'numFastSims': 20,
    'probFastSim': 0.75,
    'compareWithBaseline': True,
    'arenaCompare': 128,
    'arena_batch_size': 16,
    'arenaTemp': 0.5,
    'arenaMCTS': True,
    'baselineCompareFreq': 1,
    'model_gating': True,
    'compareWithPast': True,
    'pastCompareFreq': 1,
    'cpuct': 3,
    'fpu_reduction': 0,
    'load_model': True,
    'nnet_type':'graphnet',
    'constant_edges':True,
    'middle_layers':[256]
}),
    max_gating_iters=None,
    max_moves=MAX_TURNS,

    lr=0.01,
    num_channels=128,
    depth=3,
    value_head_channels=64,
    policy_head_channels=64,
)
args.scheduler_args.milestones = [75, 150]
"""
def doTrain():
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()


if __name__ == "__main__":
    #cProfile.runctx("doTrain()", globals(), locals(), "trainhex_5x5_observing_5x5x4_basicSym.prof")
    doTrain()

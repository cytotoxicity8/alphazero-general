from pyximport import install as pyxinstall
from numpy import get_include
pyxinstall(setup_args={'include_dirs': get_include()})

from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.utils import get_iter_file, dotdict, get_game_results, default_temp_scaling, default_const_args
from alphazero.Arena import Arena
from alphazero.GenericPlayers import RawMCTSPlayer, NNPlayer, MCTSPlayer
from alphazero.pytorch_classification.utils import Bar, AverageMeter

from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tensorboardX import SummaryWriter
from glob import glob
from queue import Empty
from time import time
from math import ceil, floor
from enum import Enum

import numpy as np
import torch
import pickle
import os
import itertools

class ModeOfGameGen(Enum):
    # Where n = numPlayers; 
    
    # For all i1, ..., in there will be an equal number of games (i1,...,in)
    CROSS_PRODUCT  = 0
    # Each worker will uniformly at random get assigned an i1, ..., in  and they will compute only those games
    ONE_PER_WORKER = 1

DEFAULT_ARGS = dotdict({
    'run_name': 'boardgame',
    'cuda': torch.cuda.is_available(),
    'workers': mp.cpu_count(),
    'startIter': 0,
    'numIters': 1000,
    'process_batch_size': 256,
    'train_batch_size': 1024,
    'arena_batch_size': 64,
    'train_steps_per_iteration': 64,
    'train_sample_ratio': 1,
    'averageTrainSteps': False,
    'autoTrainSteps': True,   # Calculates the average number of samples in the training window
                              # if averageTrainSteps set to True, otherwise uses the latest
                              # number of samples, and does train_sample_ratio * avg_num_steps 
                              # or last_num_train_steps // train_batch_size
                              # training steps.
    'train_on_past_data': False,
    'past_data_chunk_size': 25,
    'past_data_run_name': 'boardgame',
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 256 * mp.cpu_count(),
    'minTrainHistoryWindow': 4,
    'maxTrainHistoryWindow': 20,
    'trainHistoryIncrementIters': 2,
    '_num_players': None,  # Doesn't have to be changed, set automatically by the env.
    'min_discount': 1,
    'fpu_reduction': 0.2,
    'num_stacked_observations': 1,  # TODO: built-in stacked observations (arg does nothing right now)
    'numWarmupIters': 1,  # Iterations where games are played randomly, 0 for none
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    # This will let the MCTS search know that you are using canonical states (i.e all
    #   states are from the perspective of one player). It will then interpret the value
    #   vector returned from the neural network differently. It will go from :
    #     (ℙ(Player 1 wins), ℙ(Player 2 wins),ℙ(Draw)) to
    #     (ℙ(Player about to play wins), ℙ(Other player wins),ℙ(Draw))
    # Note this shouldn't affect the output of win_state, however should change the output
    #  of symmetry (i.e. when you call game.win_state() you should still get [0,1,0] if Player
    #  2 wins but when you call symmetry you should get [1,0,0] if player 2 wins and it's player
    #  2's turn
    'mctsCanonicalStates': False,
    'policy_softmax_temperature': 1.4,
    'value_softmax_temperature': 1.4,

    # Weather to use population to train hyperparameters
    'withPopulation' : False,
    # See the above enum - defines what games each worker will compute in self play
    'modeOfAssigningWork' : ModeOfGameGen.ONE_PER_WORKER,
    # The number of individuals in the population
    'populationSize' : 1,
    # A function that given an id of a memeber of the population (from 0 to populationSize - 1)
    #  that returns the trainable arguments to start that instance on
    'getInitialArgs' : default_const_args,
    # How often a roundRobin is done to optimize hyperparameters
    'roundRobinFreq' : 5,
    # If true then the round robin will take place by replacing a self play session
    #  every roundRobinFreq number of steps
    'roundRobinAsSelfPlay' : True,
    # Ignored if roundRobinAsSelfPlay
    # Number of games played between each net in the round robin
    #  note populationSize^2 * (num below) games will be played in total
    'roundRobinGames' : 6,
    # The %age of the nets that will be killed
    #  They will be from the bottom of the round robin
    'percentageKilled' : 0.2,
    # Which net to use for compare to baseline and elo
    #  If none will start with a round robin to calc best
    'bestNet' : None,
    # Defines the max deviation from original values that can happen when a new models is created
    'deviation' : 0.2,


    'symmetricSamples': True,
    'numMCTSSims': 100,
    'numFastSims': 20,
    'numWarmupSims': 5,
    'probFastSim': 0.75,
    'mctsResetThreshold': None,
    'startTemp': 1,
    'temp_scaling_fn': default_temp_scaling,
    'root_policy_temp': 1.1,
    'root_noise_frac': 0.1,
    'add_root_noise': True,
    'add_root_temp': True,
    'compareWithBaseline': True,
    'baselineTester': RawMCTSPlayer,
    'arenaCompareBaseline': 128,
    'arenaCompare': 128,
    'eloMCTS': 15,
    'eloGames':10,
    'eloMatches':10,
    'eloUniform': False,
    'calculateElo': True,
    'arenaTemp': 0.25,
    'arenaMCTS': True,
    'arenaBatched': True,
    'baselineCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 1,
    'model_gating': True,
    'max_gating_iters': None,
    'min_next_model_winrate': 0.52,
    'use_draws_for_winrate': True,
    'load_model': True,
    'cpuct': 1.25,
    'value_loss_weight': 1.5,
    'checkpoint': 'checkpoint',
    'data': 'data',

    'scheduler': torch.optim.lr_scheduler.MultiStepLR,
    'scheduler_args': dotdict({
        'milestones': [75, 125],
        'gamma': 0.1

        # 'min_lr': 1e-4,
        # 'patience': 3,
        # 'cooldown': 1,
        # 'verbose': False
    }),

    'lr': 1e-2,
    'optimizer': torch.optim.SGD,
    'optimizer_args': dotdict({
        'momentum': 0.9,
        'weight_decay': 1e-4
    }),

    'nnet_type': 'resnet',  # 'resnet' or 'fc'
    'num_channels': 32,
    'depth': 4,
    'value_head_channels': 16,
    'policy_head_channels': 16,

    # fc only uses the following
    'input_fc_layers': [1024] * 4,  # only for fc networks
    'value_dense_layers': [1024, 512],
    'policy_dense_layers': [1024, 512]
})


def get_args(args=None, **kwargs):
    new_args = DEFAULT_ARGS
    if args:
        new_args.update(args)
    for key, value in kwargs.items():
        setattr(new_args, key, value)

    if new_args.mctsCanonicalStates:
        assert new_args.symmetricSamples, "Counting who has won with cannonical state representation of board requires symetries to get win_state into correct form"

    assert new_args.model_gating, "Currently there is a silent bug without Model Gating so don't use unless searching for that bug"

    return new_args

class TrainState(Enum):
    STANDBY = 0
    INIT = 1
    INIT_AGENTS = 2
    SELF_PLAY = 3
    SAVE_SAMPLES = 4
    PROCESS_RESULTS = 5
    KILL_AGENTS = 6
    TRAIN = 7
    COMPARE_BASELINE = 8
    COMPARE_PAST = 9
    ROUND_ROBIN = 10


def _set_state(state: TrainState):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self.state = state
            ret = func(self, *args, **kwargs)
            self.state = TrainState.STANDBY
            return ret
        return wrapper
    return decorator


class Coach:
    @_set_state(TrainState.INIT)
    def __init__(self, game_cls, nnet, args):
        np.random.seed()
        self.game_cls = game_cls
        self.numNets = args.populationSize if args.withPopulation else 1
        self.train_nets = np.full(self.numNets, None)
        self.self_play_nets = np.full(self.numNets, None)
        
        self.elo_play_net = nnet.__class__(game_cls, args)
        self.elo_play_net_2 = nnet.__class__(game_cls, args)
        self.args = args
        self.args._num_players = self.game_cls.num_players() + self.game_cls.has_draw()
        self.args.bestNet  = self.args.bestNet if args.withPopulation else 0

        for i in range(0, self.numNets):
            argsi = args.copy()
            #print(args.getInitialArgs(i))
            argsi.update(args.getInitialArgs(i))

            self.train_nets[i]      = nnet.__class__(game_cls, argsi)
            self.self_play_nets[i]  = nnet.__class__(game_cls, argsi)

        
        train_iter = self.args.startIter
        self.trainableArgs = None if not(args.withPopulation) else set(self.args.getInitialArgs(0).keys())
        self.argsNotToCheck = {'startIter'}
        if self.args.load_model:
            networks = sorted(glob(self.args.checkpoint + '/' + self.args.run_name + '/*'))
            self.args.startIter = len(networks)//self.numNets
            if self.args.startIter == 0:
                for net in range(0, self.numNets):
                    self._save_model(self.train_nets[net], 0, net)
                self.args.startIter = 1

            train_iter = self.args.startIter - 1
            for net in range(0, self.numNets): 
                self._load_model(self.train_nets[net], train_iter, net)
            del networks

        self.self_play_iter = np.full(self.numNets, 0)

        for net in range(0, self.numNets):
            if self.train_nets[i].args.selfPlayModelIter == 0:
                self.self_play_iter[net] = 0
            else:
                self.self_play_iter[net] = self.train_nets[i].args.selfPlayModelIter or train_iter

        if self.args.model_gating:
            for net in range(0, self.numNets):
                self._load_model(self.self_play_nets[net], self.self_play_iter[net],net)
        

        self.gating_counter = np.zeros(self.numNets)
        self.warmup = False
        self.loss_pis = np.zeros(self.numNets)
        self.loss_vs = np.zeros(self.numNets)
        self.sample_time = 0
        self.iter_time = 0
        self.eta = 0
        self.arena = None
        self.model_iter = self.args.startIter
        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.games_for_agent = []
        self.stop_train = mp.Event()
        self.pause_train = mp.Event()
        self.stop_agents = mp.Event()
        for net in range(0, self.numNets):
            self.train_nets[net].stop_train = self.stop_train
            self.train_nets[net].pause_train = self.pause_train
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)
        if self.args.run_name != '':
            self.writer = SummaryWriter(log_dir='runs/' + self.args.run_name)
        else:
            self.writer = SummaryWriter()
        # self.args.expertValueWeight.current = self.args.expertValueWeight.start

    # Returns smaller dot dictionary of all attributes that can be learned and their vals in current dict
    def get_trainable_attributes(self, args : dotdict) -> dotdict:
        return dotdict({key: args[key] for key in self.trainableArgs})

    def _load_model(self, model, iteration, i):
        modelArgs = model.load_checkpoint(
            folder=os.path.join(self.args.checkpoint, self.args.run_name),
            filename=get_iter_file(iteration, i)
        )
        
        # Check that it agrees not on trainable args if training more than one
        if self.numNets != 1:
            for key in modelArgs:
                val = self.args[key]
                assert ((self.args[key] == val) or (key in self.trainableArgs) or (key in self.argsNotToCheck)), "One of the models differs from default arguments not on a training arguement - {} is {} in the model but {} in the defaults".format(key, val, self.args.get(key))

    def _save_model(self, model, iteration, i):
        model.save_checkpoint(
            folder=os.path.join(self.args.checkpoint, self.args.run_name),
            filename=get_iter_file(iteration, i)
        )

    def learn(self):
        print('Because of batching, it can take a long time before any games finish.')

        try:
            print("-----args-------")
            for i,j in self.args.items():
                print("\"{}\" : {},".format(i,j))
            while self.model_iter <= self.args.numIters:
                print(f'------ITER {self.model_iter}------')

                if ((not self.args.skipSelfPlayIters\
                        or self.model_iter > self.args.skipSelfPlayIters)\
                    and not (self.args.train_on_past_data and self.model_iter == self.args.startIter)):
                    if self.model_iter <= self.args.numWarmupIters:
                        print('Warmup: random policy and value')
                        self.warmup = True
                    # elif self.self_play_iter == 0:
                    #     self.warmup = True
                    elif self.warmup:
                        self.warmup = False

                    if self.warmup:
                        print('Warmup: random policy and value')

                    reset = None
                    
                    if self.args.withPopulation and\
                       self.args.roundRobinAsSelfPlay and\
                       ((self.model_iter - 1) %self.args.roundRobinFreq == 0 or\
                        self.args.bestNet == None):
                       reset = self.args.modeOfAssigningWork, self.args. arenaTemp
                       self.args.modeOfAssigningWork = ModeOfGameGen.CROSS_PRODUCT

                    self.generateSelfPlayAgents(exact = (reset != None))
                    self.processSelfPlayBatches(self.model_iter)
                    if self.stop_train.is_set():
                        break
                    self.saveIterationSamples(self.model_iter)
                    if self.stop_train.is_set():
                        break
                    dat = self.processGameResults(self.model_iter)

                    if reset != None:
                        self.args.modeOfAssigningWork = reset
                        self.roundRobin(self.model_iter-1, dat[0], dat[1])

                    if self.stop_train.is_set():
                        break
                    self.killSelfPlayAgents()
                    if self.stop_train.is_set():
                        break

                self.train(self.model_iter)
                if self.stop_train.is_set():
                    break
                
                if self.args.compareWithBaseline and (self.model_iter - 1) % self.args.baselineCompareFreq == 0:
                    net = self.args.bestNet
                    self.compareToBaseline(self.model_iter, net)
                    if self.stop_train.is_set():
                        break

                if self.args.compareWithPast and (self.model_iter - 1) % self.args.pastCompareFreq == 0:
                    for net in range(0, self.numNets):
                        self.compareToPast(self.model_iter, net)
                    if self.stop_train.is_set():
                        break

                # if (self.model_iter % 3 == 0 and self.model_iter > 1):
                #     self.tuneHyperparams(5)
                if self.args.calculateElo:
                    net = self.args.bestNet
                    self.calculateElo(net)

                for net in range(0, self.numNets):
                    self.writer.add_scalar(str(net)+'win_rate/self_play_model', self.self_play_iter[net], self.model_iter)
                self.model_iter += 1
                print()

        except KeyboardInterrupt:
            pass

        print()
        self.writer.close()
        if self.agents:
            self.killSelfPlayAgents()

    # Tries to evenly divide up all the different combinations of self play games 
    #  between all the workers
    def gamesFor(self, i):
        if self.args.modeOfAssigningWork == ModeOfGameGen.CROSS_PRODUCT:
            numPlayers = self.game_cls.num_players()
            numWorkers = self.args.workers
            numPerPair = round(self.args.gamesPerIteration/(self.numNets**numPlayers))
            lists = list(itertools.product(list(range(0, self.numNets)), repeat = numPlayers))
     
            step = len(lists)//numWorkers
            rem  = len(lists)%numWorkers
        
            retList = lists[floor(step*i) : floor(step*(i+1))] + ([] if (rem == 0) else [lists[-1-floor(rem/numWorkers * i)]])

            return (numPerPair, retList)

        elif self.args.modeOfAssigningWork == ModeOfGameGen.ONE_PER_WORKER:
            nets = np.array(range(self.numNets))

            ret = [np.random.choice(nets) for _ in range(self.game_cls.num_players())]
            return (self.args.gamesPerIteration, [tuple(ret)])

        else:
            raise ValueError("modeOfAssigningWork must be set to an element of ModeOfGameGen (or the mode you have picked is not implemented)")
 
    @_set_state(TrainState.INIT_AGENTS)
    def generateSelfPlayAgents(self, exact = False):
        self.stop_agents = mp.Event()
        self.ready_queue = mp.Queue()    
        for i in range(self.args.workers):
            self.input_tensors.append(torch.zeros(
                [self.args.process_batch_size, *self.game_cls.observation_size()]
            ))
            self.input_tensors[i].share_memory_()

            self.policy_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game_cls.action_size()]
            ))
            self.policy_tensors[i].share_memory_()

            self.value_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game_cls.num_players() + 1]
            ))
            self.value_tensors[i].share_memory_()
            self.batch_ready.append(mp.Event())

            if self.args.cuda:
                self.input_tensors[i].pin_memory()
                self.policy_tensors[i].pin_memory()
                self.value_tensors[i].pin_memory()

            #print(self.gamesFor(i))
            self.games_for_agent.append(self.gamesFor(i))
            self.agents.append(
                SelfPlayAgent(i, self.games_for_agent[i], self.game_cls, self.ready_queue, self.batch_ready[i],
                              self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                              self.result_queue, self.completed, self.games_played, self.stop_agents, self.pause_train,
                              self.args, _is_warmup=self.warmup, _exact_game_count=exact)
            )
            self.agents[i].daemon = True
            self.agents[i].start()

    @_set_state(TrainState.SELF_PLAY)
    def processSelfPlayBatches(self, iteration):
        sample_time = AverageMeter()
        bar = Bar('Generating Samples', max=self.args.gamesPerIteration)
        end = time()
        nnets = self.self_play_nets if self.args.model_gating else self.train_nets
        n = 0
        while self.completed.value != self.args.workers:
            if self.stop_train.is_set() and not self.stop_agents.is_set():
                self.stop_agents.set()

            try:
                id, netsNumbers, roundNum = self.ready_queue.get(timeout=1)
                indexToNet = self.games_for_agent[id][1]
                cumulative = 0
                for i in range(0, len(netsNumbers)):
                    if netsNumbers[i] == 0:
                        continue;
                    policy, value = nnets[indexToNet[i][roundNum]].process(self.input_tensors[id][cumulative:cumulative+netsNumbers[i]])
                    self.policy_tensors[id][cumulative: cumulative + netsNumbers[i]].copy_(policy)
                    self.value_tensors[id][cumulative: cumulative + netsNumbers[i]].copy_(value)
                    cumulative += netsNumbers[i]
                
                self.batch_ready[id].set()
            except Empty:
                pass

            size = self.games_played.value
            if size > n:
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()
            bar.suffix = f'({size}/{self.args.gamesPerIteration}) Sample Time: {sample_time.avg:.3f}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.goto(size)
            self.sample_time = sample_time.avg
            self.iter_time = bar.elapsed_td
            self.eta = bar.eta_td

        if not self.stop_agents.is_set(): self.stop_agents.set()
        bar.update()
        bar.finish()
        self.writer.add_scalar('loss/sample_time', sample_time.avg, iteration)
        print()

    @_set_state(TrainState.SAVE_SAMPLES)
    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
        print(f'Saving {num_samples} samples')

        data_tensor = torch.zeros([num_samples, *self.game_cls.observation_size()])
        policy_tensor = torch.zeros([num_samples, self.game_cls.action_size()])
        value_tensor = torch.zeros([num_samples, self.game_cls.num_players() + 1])
        for i in range(num_samples):
            data, policy, value = self.file_queue.get()
            data_tensor[i] = torch.from_numpy(data)
            policy_tensor[i] = torch.from_numpy(policy)
            value_tensor[i] = torch.from_numpy(value)

        folder = os.path.join(self.args.data, self.args.run_name)
        filename = os.path.join(folder, get_iter_file(iteration).replace('.pkl', ''))
        if not os.path.exists(folder): os.makedirs(folder)

        torch.save(data_tensor, filename + '-data.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(policy_tensor, filename + '-policy.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(value_tensor, filename + '-value.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        del data_tensor
        del policy_tensor
        del value_tensor

    @_set_state(TrainState.PROCESS_RESULTS)
    def processGameResults(self, iteration):
        num_games = self.result_queue.qsize()
        wins, draws, avg_game_length = get_game_results(self.numNets, self.result_queue, self.game_cls)
        
        numWins = np.sum(wins, axis = tuple(range(len(wins.shape) - 1)))
        numDraws = np.sum(draws)
        numAvgGameLength = np.sum(avg_game_length)

        numPlayers = self.game_cls.num_players()
        print(numWins)
        print(numDraws)
        print(numAvgGameLength)

        for i in range(numPlayers):
            self.writer.add_scalar(f'win_rate/player{i}', (
                    numWins[i] + (numDraws/numPlayers if self.args.use_draws_for_winrate else 0))/ 
                    num_games, iteration)
        self.writer.add_scalar('win_rate/draws', numDraws / num_games, iteration)
        self.writer.add_scalar('win_rate/avg_game_length', numAvgGameLength, iteration)
        
        return wins, draws, numAvgGameLength

    @_set_state(TrainState.KILL_AGENTS)
    def killSelfPlayAgents(self):
        # clear queues to prevent deadlocking
        for _ in range(self.ready_queue.qsize()):
            try:
                self.ready_queue.get_nowait()
            except Empty:
                break
        for _ in range(self.file_queue.qsize()):
            try:
                self.file_queue.get_nowait()
            except Empty:
                break
        for _ in range(self.result_queue.qsize()):
            try:
                self.result_queue.get_nowait()
            except Empty:
                break

        for agent in self.agents:
            agent.join()
            del self.input_tensors[0]
            del self.policy_tensors[0]
            del self.value_tensors[0]
            del self.batch_ready[0]

        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.games_for_agent = []
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)

    @_set_state(TrainState.TRAIN)
    def train(self, iteration):
        num_train_steps = 0
        sample_counter = 0
        
        def add_tensor_dataset(train_iter, tensor_dataset_list, run_name=self.args.run_name):
            filename = os.path.join(
                os.path.join(self.args.data, run_name), get_iter_file(train_iter).replace('.pkl', '')
            )
            
            try:
                data_tensor = torch.load(filename + '-data.pkl')
                policy_tensor = torch.load(filename + '-policy.pkl')
                value_tensor = torch.load(filename + '-value.pkl')
            except FileNotFoundError as e:
                print('Warning: could not find tensor data. ' + str(e))
                return
            
            tensor_dataset_list.append(
                TensorDataset(data_tensor, policy_tensor, value_tensor)
            )
            nonlocal num_train_steps
            if self.args.averageTrainSteps:
                nonlocal sample_counter
                num_train_steps += data_tensor.size(0)
                sample_counter += 1
            else:
                num_train_steps = data_tensor.size(0)

        def train_data(tensor_dataset_list, train_on_all=False):
            dataset = ConcatDataset(tensor_dataset_list)
            dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                    num_workers=self.args.workers, pin_memory=True)
            if self.args.averageTrainSteps:
                nonlocal num_train_steps
                num_train_steps //= sample_counter
            train_steps = len(dataset) // self.args.train_batch_size \
               if train_on_all else (num_train_steps // self.args.train_batch_size
                   if self.args.autoTrainSteps else self.args.train_steps_per_iteration)
            result = np.zeros([2,self.numNets])
            for toTrain in range(0, self.numNets):
                 result[0][toTrain], result[1][toTrain] = self.train_nets[toTrain].train(dataloader, train_steps)

            del dataloader
            del dataset

            return result

        if self.args.train_on_past_data and iteration == self.args.startIter:
            next_start_iter = 1
            total_iters = len(
                glob(os.path.join(os.path.join(self.args.data, self.args.past_data_run_name), '*.pkl'))
            ) // 3
            num_chunks = ceil(total_iters / self.args.past_data_chunk_size)
            print(f'Training on past data from run "{self.args.past_data_run_name}" in {num_chunks} chunks of '
                  f'{self.args.past_data_chunk_size} iterations ({total_iters} iterations in total).')

            for _ in range(num_chunks):
                datasets = []
                i = next_start_iter
                for i in range(next_start_iter, min(
                    next_start_iter + self.args.past_data_chunk_size, total_iters + 1
                )):
                    add_tensor_dataset(i, datasets, run_name=self.args.past_data_run_name)
                next_start_iter = i + 1

                self.loss_pis, self.loss_vs = train_data(datasets, train_on_all=True)
                del datasets
        else:
            datasets = []

            # current_history_size = self.args.numItersForTrainExamplesHistory
            current_history_size = min(
                max(
                    self.args.minTrainHistoryWindow,
                    (iteration + self.args.minTrainHistoryWindow) // self.args.trainHistoryIncrementIters
                ),
                self.args.maxTrainHistoryWindow
            )

            [add_tensor_dataset(i, datasets) for i in range(max(1, iteration - current_history_size), iteration + 1)]
            self.loss_pis, self.loss_vs = train_data(datasets)

        net = self.args.bestNet
        self.writer.add_scalar('loss/policy', self.loss_pis[net], iteration)  # TODO: policy loss not showing up in tensorboard
        self.writer.add_scalar('loss/value', self.loss_vs[net], iteration)
        self.writer.add_scalar('loss/total', self.loss_pis[net] + self.loss_vs[net], iteration)

        for net in range(self.numNets):
            self._save_model(self.train_nets[net], iteration, net)

    def calculateElo(self, player):
        if not os.path.exists("elo/"+self.args.run_name):
            os.makedirs("elo/"+self.args.run_name)
        networks = sorted(glob(self.args.checkpoint + '/' + self.args.run_name + '/*'))
        if self.model_iter == 1:
            np.savetxt('elo/'+self.args.run_name+'/' + 'ELOS.csv', [[0]], delimiter=",")

        elos = np.loadtxt('elo/'+self.args.run_name+'/' + 'ELOS.csv', delimiter=',')
        elos = [elos]
        elos = np.array(elos).flatten()
        # print(elos)
        # current_elo = elos[len(elos)-1]
        current_elo = 0

        sf_args = self.args.copy()
        sf_args.numMCTSSims = self.args.eloMCTS
        cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
        nplayer = cls(self.train_nets[player], self.game_cls, sf_args)
        running_score = 0
        running_expectation = 0

        #Sample from harmonic distribution because integral calculus of 
        #normal distribution is nasty and i dont like the error function
        #Not used, but testing it out. Currently it is just a uniform sampling
        def harmonic(n):
            a = 0
            for i in range(0, n):
                a += 1/(1+i)
            return a

        num_games = self.args.eloMatches
        harmonic_coef = 1/harmonic(len(elos))
        probs = harmonic_coef / (1+np.arange(0, len(elos)))

        #opponents = np.random.choice(np.flip(np.arange(0, len(elos))), p=probs, size=num_games)
        
        opponents = np.random.choice(np.arange(0, len(elos)), size=num_games)
        if self.args.eloUniform:
            opponents = np.arange(max(0, len(elos)-num_games), len(elos))
        print(f"Pitting against the following iters:{opponents}")
        for i in opponents:
            print(f'PITTING AGAINST ITERATION {i} FOR ELO CALCULATION ')
            opponent_elo = elos[i]
            self._load_model(self.elo_play_net, i, player)
            pplayer = cls(self.elo_play_net, self.game_cls, sf_args)
            players = [nplayer] + [pplayer] * (self.game_cls.num_players() - 1)
            self.arena = Arena(players, self.game_cls, use_batched_mcts=False, args=sf_args)
            wins, draws, winrates = self.arena.play_games(self.args.eloGames, verbose=False)

            expected_score = 1/(1+10**( (opponent_elo-current_elo)/400 ))
            actual_score = (wins[0] + 0.5*draws)#/(wins[0]+wins[1]+draws)
            running_expectation += 10*expected_score
            running_score += actual_score
            #current_elo = current_elo + 32*(actual_score - 10*expected_score)

        current_elo = current_elo + 32*(running_score - running_expectation)
        current_elo = max(current_elo, 0)
        elos = np.append(elos, current_elo)
        np.savetxt('elo/'+self.args.run_name+'/' + 'ELOS.csv', [elos], delimiter=",")
        print(f'Self play ELO : {current_elo}')
        self.writer.add_scalar('elo/self_play_elo', current_elo, self.model_iter)

    def randomPreviousGames(self, ITER):
        elos = np.loadtxt('elo/'+self.args.run_name+'/ELOS.csv', delimiter=',')
        elos = [elos]
        elos = np.array(elos).flatten()
        for i in range(1, len(elos)):
            #print(i, elos[i])
            self.writer.add_scalar('elo/self_play_elo_3', elos[i], i)
    
    def sweepCPUCT(self, num):
        params = np.linspace(0.25, 5, num)

        self._load_model(self.elo_play_net, self.model_iter)
        cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer

        bestIndex = -1
        highestRate = 0
        for i in range(num):
            print(f"Testing CPUCT: {params[i]}")
            new_args = self.args.copy()
            new_args.cpuct = params[i]
            nplayer = cls(self.elo_play_net, self.game_cls, new_args)
            pplayer = cls(self.elo_play_net, self.game_cls, self.args)
            players = [nplayer] + [pplayer] * (self.game_cls.num_players() - 1)
            self.arena = Arena(players, self.game_cls, use_batched_mcts=False, args=self.args)
            wins, draws, wrs = self.arena.play_games(10, verbose=False)
            if wrs[0] > highestRate and wrs[0] > 0.52:
                highestRate = wrs[0]
                bestIndex = i

        if bestIndex != -1:
            print(f"Optimimum CPUCT: {params[bestIndex]}")
            self.args.cpuct = params[bestIndex]
        self.writer.add_scalar("hyperparmeters/CPUCT", self.args.cpuct, self.model_iter)

    #Testing code--- Not working for the moment
    def tuneHyperparams(self, num):
        print()
        print(f"Tuning hyperparmeters with population size of {num}")
        if not os.path.exists("hyperparams/"+self.args.run_name):
            os.makedirs("hyperparams/"+self.args.run_name)
        if self.model_iter == 1:
            np.savetxt("hyperparams/"+self.args.run_name+"/params.csv", [[self.args.cpuct]], delimiter=",")

        recent = np.loadtxt('hyperparams/'+self.args.run_name+'/params.csv', delimiter=',')
        recent = [recent]
        recent = np.array(recent).flatten()
        # Loaded previous hyperparameters
        print(f"Loading most recent CPUCT: {recent}")
        new_args = self.args.copy()
        new_args.cpuct = recent[0]
        params = [new_args.copy() for i in range(num)]
        WINRATES = [0] * num
        RANGE = 0.35
        #Mutate some params
        params[0].numMCTSSims = 15
        for i in range(1, len(params)):
            # params[i].fpu_reduction = np.clip(params[i].fpu_reduction + params[i].fpu_reduction * np.random.uniform(-RANGE, RANGE), 0, 1)
            params[i]["cpuct"] = np.clip(params[i].cpuct + np.random.uniform(-RANGE, RANGE), 0.25, 5)
            # params[i].root_policy_temp = params[i].root_policy_temp + params[i].root_policy_temp * np.random.uniform(-RANGE, RANGE)
            # params[i].root_noise_frac = params[i].root_noise_frac + params[i].root_noise_frac * np.random.uniform(-RANGE, RANGE)
            params[i].numMCTSSims = 15
            #print(params[i].fpu_reduction, params[i].cpuct, params[i].root_policy_temp, params[i].root_noise_frac)
        #Round robin
        for i in range(len(params)):
            print(params[i].cpuct)
        self._load_model(self.elo_play_net, self.model_iter, player)
        cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
        
        for p in range(len(params)):
            for o in range(len(params)):
                if p != o:
                    print(f"Pitting {p} against {o} with CPUCT: {params[p].cpuct} and {params[o].cpuct}. {((p)*num+(o))/(num*num) * 100}% Complete")
                    nplayer = cls(self.elo_play_net, self.game_cls, params[p])
                    pplayer = cls(self.elo_play_net, self.game_cls, params[o])
                    players = [nplayer] + [pplayer] * (self.game_cls.num_players() - 1)
                    self.arena = Arena(players, self.game_cls, use_batched_mcts=False, args=self.args)
                    wins, draws, wrs = self.arena.play_games(6, verbose=False)
                    WINRATES[p] += wrs[0]

        best = np.argmax(WINRATES)
        recent[0] = params[best].cpuct
        print("Optimimum Found:")
        print(f"CPUCT: {params[best].cpuct}")
        self.args = params[best].copy()
        np.savetxt("hyperparams/"+self.args.run_name+"/params.csv", [recent], delimiter=",")
        # self.writer.add_scalar("hyperparmeters/FPU", params[best].fpu_reduction, self.model_iter)
        self.writer.add_scalar("hyperparmeters/CPUCT", params[best].cpuct, self.model_iter)
        # self.writer.add_scalar("hyperparmeters/ROOT_POLICY_TEMP", params[best].root_policy_temp, self.model_iter)
        # self.writer.add_scalar("hyperparmeters/ROOT_NOISE_FRAC", params[best].root_noise_frac, self.model_iter)
    
    @_set_state(TrainState.ROUND_ROBIN)
    def roundRobin(self, iteration, wins, draws):
        print('PERFORMING ROUND ROBIN ANALYSIS')
        print()
        totalWins  = np.zeros(self.numNets)
        numPlayers = self.game_cls.num_players()

        for win, numWins in np.ndenumerate(wins):
            if numWins == 0:
                continue;
            whoWon  = win[-1]
            totalWins[win[whoWon]] += numWins
        
        if self.args.use_draws_for_winrate:
            for draw, numDraws in np.ndenumerate(draws):
                if numDraws == 0:
                    continue;
                for p in draw:
                    totalWins[p] += numDraws/numPlayers

        ranking = np.argsort(totalWins)

        self.args.bestNet = ranking[0]

        numReplace = round(self.args.percentageKilled * self.numNets)
        
        for replacerIndex in range(0, numReplace):
            replacingIndex = self.numNets - replacerIndex - 1

            replacerNet  = ranking[replacerIndex]
            replacingNet = ranking[replacingIndex]

            print(f'REPLACING {replacingNet} WITH {replacerNet}')
            argsToUpdate = self.get_trainable_attributes(self.train_nets[replacerNet].args)
            for key in argsToUpdate:
                argsToUpdate[key] = argsToUpdate[key] * np.random.uniform(1-self.args.deviation, 1+self.args.deviation)

            
            self._load_model(self.train_nets[replacingNet], iteration, replacerNet)
            self.train_nets[replacingNet].args.update(argsToUpdate)


            if self.args.model_gating:
                self.self_play_iter[replacingNet] = self.self_play_iter[replacerNet]
                self._load_model(self.self_play_nets[replacingNet], self.self_play_iter[replacingNet], replacerNet)
                self.self_play_nets[replacingNet].args.update(argsToUpdate)

        print('NEW ARGUEMENTS ARE : ')
        for net in range(self.numNets):
            print(f"{net} : {self.get_trainable_attributes(self.train_nets[net].args)}")

        print(f'NEW BEST NET IS {self.args.bestNet}')
        



    @_set_state(TrainState.COMPARE_PAST)
    def compareToPast(self, model_iter, player):
        self._load_model(self.self_play_nets[player], self.self_play_iter[player], player)

        print(f'PITTING P{player} AGAINST ITERATION {self.self_play_iter[player]}')
        # if self.args.arenaBatched:
        #     if not self.args.arenaMCTS:
        #         self.args.arenaMCTS = True
        #         print('WARNING: Batched arena comparison is enabled which uses MCTS, but arena MCTS is set to False.'
        #                           ' Ignoring this, and continuing with batched MCTS in arena.')

        #     nplayer = self.train_net.process
        #     pplayer = self.self_play_net.process
        # else:
        #     cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
        #     nplayer = cls(self.game_cls, self.args, self.train_net)
        #     pplayer = cls(self.game_cls, self.args, self.self_play_net)
        
        cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
        nplayer = cls(self.train_nets[player], self.game_cls, self.args)
        pplayer = cls(self.self_play_nets[player], self.game_cls, self.args)

        players = [nplayer] + [pplayer] * (self.game_cls.num_players() - 1)
        self.arena = Arena(players, self.game_cls, use_batched_mcts=self.args.arenaBatched, args=self.args)
        wins, draws, winrates = self.arena.play_games(self.args.arenaCompare)
        if self.stop_train.is_set(): return
        winrate = winrates[0]

        print(f'NEW/PAST WINS FOR {player} : {wins[0]} / {sum(wins[1:])} ; DRAWS : {draws}\n')
        print(f'NEW MODEL WINRATE {player} : {round(winrate, 3)}')
        if player == self.args.bestNet:
            self.writer.add_scalar('win_rate/past', winrate, model_iter)

        ### Model gating ###
        if (
            self.args.model_gating
            and winrate < self.args.min_next_model_winrate
            and (self.args.max_gating_iters is None
                 or self.gating_counter < self.args.max_gating_iters)
        ):
            self.gating_counter[player] += 1
        elif self.args.model_gating:
            print("No Gating")
            self.self_play_iter[player] = model_iter
            self._load_model(self.self_play_nets[player], self.self_play_iter[player], player)
            self.gating_counter[player] = 0

        if self.args.model_gating:
            print(f'Using model version {self.self_play_iter[player]} for P{player} self play.')

    @_set_state(TrainState.COMPARE_BASELINE)
    def compareToBaseline(self, iteration, player):
        test_player = self.args.baselineTester(self.game_cls, self.args)
        can_process = test_player.supports_process() and self.args.arenaBatched

        print()

        nnplayer = (MCTSPlayer if self.args.arenaMCTS else NNPlayer)(self.train_nets[player], self.game_cls, self.args)

        print(f'PITTING AGAINST BASELINE: ' + self.args.baselineTester.__name__)

        players = [nnplayer] + [test_player] * (self.game_cls.num_players() - 1)
        self.arena = Arena(players, self.game_cls, use_batched_mcts=self.args.arenaBatched, args=self.args)
        wins, draws, winrates = self.arena.play_games(self.args.arenaCompare)
        if self.stop_train.is_set(): return
        winrate = winrates[0]

        print(f'NEW/BASELINE WINS FOR : {wins[0]} / {sum(wins[1:])} ; DRAWS : {draws}\n')
        print(f'NEW MODEL WINRATE FOR : {round(winrate, 3)}')
        self.writer.add_scalar('win_rate/baseline', winrate, iteration)

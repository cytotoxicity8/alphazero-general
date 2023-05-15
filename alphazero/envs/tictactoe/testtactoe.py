"""
To run tests:
pytest-3 connect4
"""
import pyximport; pyximport.install()
import textwrap
import numpy as np
import cProfile
import torch_geometric as geo_torch
import networkx as nx
import matplotlib.pyplot as plt
from time import time

from .tictactoe import Game, display
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader


def init_board_from_moves(moves):
    game = Game()
    for move in moves:
        game.play_action(move)
    return game

def test_simple_moves():
    g = Game()
    for move in [0,1,2,3,4,5,6]:
        print(g.observation())
        g.play_action(move)
        display(g._board)

def draw_graph(data):
    g = geo_torch.utils.to_networkx(data, node_attrs = ['x'], to_undirected=True)
    lables = nx.get_node_attributes(g, 'x')
    nx.draw_networkx(g, labels=lables)
    plt.show()
"""
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
    
    if self.argsUsedInTraining.intersection(self.trainableArgs) == set():
        result[0][0], result[1][0] = self.train_nets[0].train(dataloader, train_steps)
        self._save_model(self.train_nets[0], iteration, 0)
        for toTrain in range(1, self.numNets):
            print("Training Net | Using other data as no args used in training are trainable - so model can be coppied")
            result[0][toTrain], result[1][toTrain] = result[0][0], result[1][0]
            tempArgs = self.train_nets[toTrain].args.copy()
            self._load_model(self.train_nets[toTrain], iteration, 0)
            self.train_nets[toTrain].args = tempArgs
            print()
    else:
        for toTrain in range(0, self.numNets):
            dataset = ConcatDataset(tensor_dataset_list)
            dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                            num_workers=self.args.workers, pin_memory=True)
    
            result[0][toTrain], result[1][toTrain] = self.train_nets[toTrain].train(dataloader, train_steps)


    del dataloader
    del dataset

    return result
"""
if __name__ == "__main__":
    
    from alphazero.GenericPlayers import *
    from alphazero.NNetWrapper import NNetWrapper as NNet
    from alphazero.envs.tictactoe.train import args, argsss

    args._num_players = 2
    #args.mctsCanonicalStates = False
    
    nn1 = NNet(Game, args)

    nn1.load_checkpoint(folder='./checkpoint/tictacgraph_5_noDropout_norm_deeper_DFS_batch', 
                        filename='00-iteration-0000.pkl')
    
    #player1 = MCTSPlayer(game_cls=Game, nn=nn1, args=args, verbose=True, draw_mcts=True, draw_depth=1)
    
    num_samples = 6

    obs_size = Game.observation_size()
    x_tensor    = torch.zeros([num_samples, obs_size[1], obs_size[0]])
    edge_tensor = torch.zeros([num_samples, 2, obs_size[2]], dtype=int)

    g = Game()
    observation1 = g.observation()
    for i in range(0,2):
        x_tensor[i]    = observation1.x
        edge_tensor[i] = observation1.edge_index
    #observation1.x = torch.FloatTensor(observation1.x.to(torch.float))

    g.play_action(4)
    observation2 = g.observation()
    for i in range(2,6):
        x_tensor[i]    = observation2.x
        edge_tensor[i] = observation2.edge_index
    #observation2.x = torch.FloatTensor(observation2.x.to(torch.float))

    vout = np.array([[0,0,1]]*num_samples)
    pout = np.array([[0,0,0,0,1,0,0,0,0]]*2+[[1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1]])
    dataset = TensorDataset(x_tensor, edge_tensor, torch.from_numpy(pout), torch.from_numpy(vout))

    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False, num_workers=1, pin_memory=True)

    #print(observation.edge_index.reshape(1,2,-1))
    #assert 1 == 0
    print(nn1.train(dataloader, 200))

    player1 = NNPlayer(game_cls=Game, nn=nn1, args=args, verbose=True)
    
    #print(observation.x)
    #print(observation.edge_index)


    #print(player1.play(g2))
    print(player1.play(g))
    #player1.reset()
    print(player1.play(Game()))
    
    
    #print(g.observation().to_dict())

    #print(g.observation().x)
    #draw_graph(g.observation())
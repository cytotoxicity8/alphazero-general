from alphazero.NNetArchitecture import ResNet, FullyConnected, GraphNet, CustomGraphModel1
from alphazero.pytorch_classification.utils import Bar, AverageMeter
from alphazero.Game import GameState
from alphazero.utils import dotdict
from threading import Event
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union


import torch.optim as optim
import numpy as np
import warnings
import torch
import torch_geometric as geo_torch
import pickle
import time
import os


class BaseWrapper(ABC):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game_cls: GameState, args):
        self.game_cls = game_cls
        self.args = args
        self.stop_train = Event()
        self.pause_train = Event()

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @abstractmethod
    def train(self, examples, num_steps: int) -> Tuple[float, float]:
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
            num_steps: the number of training steps to perform. Each step, a batch
                       is fed through the network and backpropogated.
        Returns:
            pi_loss: the average loss of the policy head during the training as a float
            val_loss: the average loss of the value head during the training as a float
        """
        pass

    @abstractmethod
    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Input:
            board: current board as a numpy array

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.action_size()
            v: a float in float range [-1,1] that gives the value of the current board
        """
        pass

    @abstractmethod
    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    @abstractmethod
    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass


class NNetWrapper(BaseWrapper):
    def __init__(self, game_cls, args):
        super().__init__(game_cls, args)
        self.nnet = None
        self._load_nnet(args)
        self.action_size = game_cls.action_size()
        self.optimizer = args.optimizer(self.nnet.parameters(), lr=args.lr, **args.optimizer_args)

        self.scheduler = args.scheduler(self.optimizer, **args.scheduler_args)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **args.scheduler_args)
        self.verbose = args.scheduler_args.get('verbose')

        if args.cuda:
            self.nnet.cuda()

        self.current_step = 0
        self.total_steps = 0
        self.l_pi = 0
        self.l_v = 0
        self.l_total = 0
        self.step_time = 0
        self.elapsed_time = 0
        self.eta = 0
        self.__loaded = False

    def _load_nnet(self, args):
        if args.nnet_type == 'resnet':
            self.nnet = ResNet(self.game_cls, args)
        elif args.nnet_type == 'fc':
            self.nnet = FullyConnected(self.game_cls, args)
        elif args.nnet_type == 'graphnet':
            self.nnet = GraphNet(self.game_cls, args)
        elif args.nnet_type == 'custom_graphmodel1':
            self.nnet = CustomGraphModel1(self.game_cls, args)
        else:
            raise ValueError(f'Unknown NNet type "{args.nnet_type}"')

    @property
    def loaded(self):
        return self.__loaded

    def train(self, batches, train_steps):
        self.total_steps = train_steps
        self.nnet.train()

        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()

        if self.verbose:
            print(f'Current LR: {self.optimizer.param_groups[0]["lr"]}')

        bar = Bar(f'Training Net', max=train_steps)
        self.current_step = 0
        while self.current_step < train_steps and not self.stop_train.is_set():
            for batch_idx, batch in enumerate(batches):
                if self.current_step == train_steps or self.stop_train.is_set():
                    break

                while self.pause_train.is_set():
                    time.sleep(.1)

                start = time.time()
                self.current_step += 1
                if self.args.nnet_type == "resnet":
                    boards, target_pis, target_vs = batch
                elif self.args.nnet_type == "graphnet":
                    xs, edges, target_pis, target_vs = batch
                    boards = geo_torch.data.Data(torch.cat([*xs]), torch.cat([*edges],-1))
                elif self.args.nnet_type == "custom_graphmodel1":
                    xs, edge_indices, edge_weights, edge_attrs, target_pis, target_vs = batch

                    tensor_edge_index = torch.cat([*edge_indices],-1)
                    tensor_edge_weight = torch.cat([*edge_weights])
                    tensor_edge_attr = torch.cat([*edge_attrs])

                    valid_edge_indices = tensor_edge_index[:, tensor_edge_weight != 0]
                    valid_edge_attrs = tensor_edge_attr[tensor_edge_weight != 0]
                    valid_edge_weights = tensor_edge_weight[tensor_edge_weight != 0]

                    #print(tensor_edge_index.shape, valid_edge_indices.shape)
                    
                    boards = geo_torch.data.Data(torch.cat([*xs]), valid_edge_indices, 
                                                 valid_edge_weights, valid_edge_attrs)

 
                    """
                    # 배치 내의 모든 그래프 데이터를 처리
                    boards = geo_torch.data.Batch.from_data_list([
                        geo_torch.data.Data(
                            x=x,
                            edge_index=ei,
                            edge_weight=ew,
                            edge_attr=ea
                        ) for x, ei, ew, ea in zip(xs, edge_indices, edge_weights, edge_attrs)
                    ])
                    """

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = (
                        boards.contiguous().cuda(),
                        target_pis.contiguous().cuda(),
                        target_vs.contiguous().cuda()
                    )

                # measure data loading time
                data_time.update(time.time() - start)
                #torch.autograd.set_detect_anomaly(True)
                # compute output
                # ONLY VALID not to exponentiate as the inputs should all be zeros except for one 1 
                out_pi, out_v = self.nnet(boards, target_vs.size()[0])
                #out_pi, out_v = torch.exp(out_pi), torch.exp(out_v)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                """
                print(target_pis)
                print(out_pi)

                print("-------")
                print(target_vs)
                print(out_v)
                """
                total_loss = l_v + l_pi
                # record loss
                pi_losses.update(l_pi.item(), target_vs.size(0))
                v_losses.update(l_v.item(), target_vs.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - start)

                self.l_pi = pi_losses.avg
                self.l_v = v_losses.avg
                self.l_total = self.l_pi + self.l_v
                self.step_time = data_time.avg + batch_time.avg
                self.elapsed_time = bar.elapsed_td
                self.eta = bar.eta_td

                # plot progress
                bar.suffix = '({step}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    step=self.current_step,
                    size=train_steps,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()

        self.scheduler.step(
            (pi_losses.avg + v_losses.avg) if isinstance(
                self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None
        )
        bar.update()  # TODO: division by zero when train steps is too small (0?)
        bar.finish()
        self.nnet.eval()
        print()

        return pi_losses.avg, v_losses.avg

    def predict(self, board: Union[np.ndarray, geo_torch.data.Data], batch_size=1):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        if self.args.nnet_type == "resnet":
            board = torch.FloatTensor(board.astype(np.float64))
        else:
            board.x = torch.FloatTensor(board.x.to(torch.float))
        
        if self.args.cuda:
            board = board.contiguous().cuda()
        with torch.no_grad():
            self.nnet.eval()
            pi, v = self.nnet(board, batch_size)

            # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
            return torch.exp(pi).data.cpu().numpy()[0], torch.exp(v).data.cpu().numpy()[0]

    def process(self, batch: Union[torch.Tensor, geo_torch.data.Batch], batch_size=1):
        if self.args.nnet_type == "resnet":
            batch = batch.type(torch.FloatTensor)
        elif self.args.nnet_type == "graphnet":
            batch.x = torch.FloatTensor(batch.x.to(torch.float))
        elif self.args.nnet_type == "custom_graphmodel1":
            batch.x = torch.FloatTensor(batch.x.to(torch.float))
        if self.args.cuda:
            batch = batch.cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(batch, batch_size)
            #print(pi)
            #print(v)

            return torch.exp(pi), torch.exp(v)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return -self.args.value_loss_weight * torch.sum(targets * outputs) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', make_dirs=True):
        filepath = os.path.join(folder, filename)
        if make_dirs and not os.path.exists(folder):
            os.makedirs(folder)

        torch.save({
            'state_dict': self.nnet.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sch_state': self.scheduler.state_dict(),
            'args': self.args
        }, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar',
                        use_saved_args=True) -> Optional[dotdict]:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))


        checkpoint = torch.load(filepath, weights_only=False)
        args_saved = 'args' in checkpoint
        if use_saved_args and args_saved:
            self.args = checkpoint['args']
            self.__init__(self.game_cls, self.args)
        elif use_saved_args and not args_saved:
            warnings.warn('No args were saved in the checkpoint file, therefore they were not loaded.')

        self.nnet.load_state_dict(checkpoint['state_dict'])
        if 'opt_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['opt_state'])
        if 'sch_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['sch_state'])

        self.__loaded = True
        if args_saved:
            return checkpoint['args']

    @classmethod
    def from_checkpoint(cls, game_cls, *args, **kwargs):
        instance = cls.__new__(cls)
        instance.game_cls = game_cls
        instance.load_checkpoint(*args, **kwargs)
        return instance

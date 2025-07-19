import numpy as np
class dotdict(dict):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        data = super().copy()
        return self.__class__(data)
    


def get_iter_file(iteration:int , number = None):
    if number != None:
        return f'{number:02d}-iteration-{iteration:04d}.pkl'
    else:
        return f'iteration-{iteration:04d}.pkl'


def scale_temp(scale_factor: float, min_temp: float, cur_temp: float, turns: int, const_max_turns: int) -> float:
    if const_max_turns and (turns + 1) % int(scale_factor * const_max_turns) == 0:
        return max(min_temp, cur_temp / 2)
    else:
        return cur_temp

def default_const_args(x):
    return dotdict({'cpuct': 1.25})


def default_temp_scaling(*args, **kwargs) -> float:
    return scale_temp(0.15, 0.2, *args, **kwargs)


def const_temp_scaling(temp, *args, **kwargs) -> float:
    return temp


def get_game_results(numberOfCompetingNets, result_queue, game_cls, _get_index=None, save_event=None):

    num_games = result_queue.qsize()
    wins = np.zeros([numberOfCompetingNets] * game_cls.num_players() + [game_cls.num_players()])
    draws = np.zeros([numberOfCompetingNets] * game_cls.num_players())
    game_len_sum = 0

    selfWins  = np.zeros([numberOfCompetingNets, game_cls.num_players(), game_cls.num_players()])
    selfDraws = np.zeros([numberOfCompetingNets, game_cls.num_players()])

    err_count = 0

    for _ in range(num_games):
        try:
            state, winstate, agent_id, players = result_queue.get()
        except:
            err_count += 1
            continue
        players = tuple(players)
        game_len_sum += state.turns

        for p in players:
            if p >= numberOfCompetingNets:
                netPos = players.index(p - numberOfCompetingNets)

                for player, is_win in enumerate(winstate):
                #print(player, is_win)
                    if is_win:
                        if player == game_cls.num_players():
                            selfDraws[players[netPos]][netPos] += 1
                        else:
                            selfWins[players[netPos]][netPos][player] += 1
                break;
        else:
            for player, is_win in enumerate(winstate):
                #print(player, is_win)
                if is_win:
                    if player == game_cls.num_players():
                        draws[players] += 1
                    else:
                        index = _get_index(player, agent_id) if _get_index else player
                        wins[players][index] += 1

    #save_event.set()
    #print(f"err_count: {err_count} / total games: {num_games}")
    return wins, draws, game_len_sum / num_games if num_games else game_len_sum, selfWins, selfDraws


def plot_mcts_tree(mcts, max_depth=2):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()

    global node_idx
    node_idx = 0

    def find_nodes(cur_node, _past_node=None, _past_i=None, _depth=0):
        if _depth > max_depth: return
        global node_idx
        cur_idx = node_idx

        G.add_node(cur_idx, a=cur_node.a, q=round(cur_node.q, 2), n=cur_node.n, v=round(cur_node.v, 2))
        if _past_node:
            G.add_edge(cur_idx, _past_i)
        node_idx += 1

        for node in cur_node._children:
            find_nodes(node, cur_node, cur_idx, _depth+1)

    find_nodes(mcts._root)
    labels = {node: '\n'.join(['{}: {}'.format(k, v) for k, v in G.nodes[node].items()]) for node in G.nodes}
    #pos = nx.spring_layout(G, k=0.15, iterations=50)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Gnodesep=1.0 -Goverlap=false')
    nx.draw(G, pos, labels=labels)
    plt.show()


def convert_checkpoint_file(filepath: str, game_cls, args: dotdict, overwrite_args=False):
    from alphazero.NNetWrapper import NNetWrapper
    nnet = NNetWrapper(game_cls, args)
    nnet.load_checkpoint('', filepath, use_saved_args=not overwrite_args)
    nnet.save_checkpoint('', filepath, make_dirs=False)


def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

import re
import hgtk

def dueum(letter):
    try:
        dec = hgtk.letter.decompose(letter)
    except:
        print("이상한 글자:", letter)
        return letter
    #제10항  한자음 ‘녀, 뇨, 뉴, 니’가 단어 첫머리에 올 적에는 두음 법칙에 따라 ‘여, 요, 유, 이’로 적는다
    if dec[0] == 'ㄴ' and dec[1] in ['ㅕ', 'ㅛ', 'ㅠ', 'ㅣ']:
        letter = hgtk.letter.compose('ㅇ', dec[1], dec[2])

    #제11항 한자음 ‘랴, 려, 례, 료, 류, 리’가 단어의 첫머리에 올 적에는 두음 법칙에 따라 ‘야, 여, 예, 요, 유, 이’로 적는다.
    elif dec[0] == 'ㄹ' and dec[1] in ['ㅑ', 'ㅕ', 'ㅖ', 'ㅛ', 'ㅠ', 'ㅣ']:
        letter = hgtk.letter.compose('ㅇ', dec[1], dec[2])

    #제12항  한자음 ‘라, 래, 로, 뢰, 루, 르’가 단어의 첫머리에 올 적에는 두음 법칙에 따라 ‘나, 내, 노, 뇌, 누, 느’로 적는다.
    elif dec[0] == 'ㄹ' and dec[1] in ['ㅏ', 'ㅐ', 'ㅗ', 'ㅚ', 'ㅜ', 'ㅡ']:
        letter = hgtk.letter.compose('ㄴ', dec[1], dec[2])
    
    return letter
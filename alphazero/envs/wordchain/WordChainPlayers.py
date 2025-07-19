from alphazero.GenericPlayers import BasePlayer
from alphazero.Game import GameState
import joblib
import pandas as pd
from alphazero.utils import dueum

#TODO: 실제 단어 쓸 수 있도록
class HumanWordChainPlayer(BasePlayer):
    ch2idx = joblib.load('alphazero/envs/wordchain/data/ch2idx.pkl')
    idx2ch = {v: k for k, v in ch2idx.items()}
    
    num_letters = len(ch2idx)
    def play(self, state: GameState, valid_data: pd.DataFrame = None) -> int:
        valid = state.valid_moves()

        while True:
            if state._board.head is not None:
                word = input(f'현재 글자는 {HumanWordChainPlayer.idx2ch[state._board.head]}입니다. 다음 글자를 제시하세요. 사용 가능한 리스트를 알고 싶으면 0을 입력하세요.')
            else:
                word = input('첫 단어를 제시하세요. 사용 가능한 리스트를 알고 싶으면 0을 입력하세요.')
            if word=='0':
                print("DB:", valid_data[valid_data["앞말"].isin([HumanWordChainPlayer.idx2ch[state._board.head], dueum(HumanWordChainPlayer.idx2ch[state._board.head])])])
                print("valid:", list(map(lambda x: HumanWordChainPlayer.idx2ch[x%HumanWordChainPlayer.num_letters], list(valid.nonzero()[0]))))
                continue
            if not self.is_word_valid_now(valid_data, state, word):
                continue

            
            try:
                if not ((valid[HumanWordChainPlayer.ch2idx[word[-1]]]==1) or (valid[HumanWordChainPlayer.ch2idx[word[-1]] + HumanWordChainPlayer.num_letters]==1)):
                    print("이거 못 씀")
                    continue
            except:
                print("이거 쓰면 짐")
                continue
            valid_data.drop(index = valid_data[valid_data["단어"] == word].index, inplace=True)

            if  HumanWordChainPlayer.ch2idx[word[0]] == state._board.head:
                word = HumanWordChainPlayer.ch2idx[word[-1]]
            else:
                word = HumanWordChainPlayer.ch2idx[word[-1]] + HumanWordChainPlayer.num_letters
            break

        return word

    def is_word_valid_now(self, valid_data:pd.DataFrame, state: GameState,word: str) -> bool:
        if not (valid_data["단어"] == word).any():
            print("현재 사용되지 않은 단어 리스트에 존재하지 않습니다.")
            return False
        if state._board.head is None:
            return True
        elif not word[0] in [HumanWordChainPlayer.idx2ch[state._board.head], dueum(HumanWordChainPlayer.idx2ch[state._board.head])]:
            print("끝말이 이어지지 않습니다.")
            return False
        else:
            return True
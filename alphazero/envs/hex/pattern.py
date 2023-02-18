description = """{-
Module      : Patern
Description : Defines what a pattern is : 
   - A collection of points to connect on a carrier with some alg of getting a 
      response given a move inside the carrier
   - If all repsonses are taken then when the carrier is full the points to connect
      will be connected
Maintainer  : River
-}"""

#-------- Imports ---------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Set
from copy import copy

#-------- Classes ---------------------------------------------------------------
class Pattern(ABC):
    @property
    @abstractmethod
    def type(self):
        pass


    @abstractmethod
    def reply(self, cords : Tuple[int,int]) -> Tuple[int,int]:
        pass;



# Defines a pairing pattern in Hex
#   https://www.hexwiki.net/index.php/Pairing_strategy
class PairingPattern(Pattern):
    type = "pairingPattern"    
    def __init__(self, type1 : str, toConnect : Set[Tuple[int,int]], 
                    pairs : List[Tuple[Tuple[int,int], Tuple[int,int]]], findCC: Callable):
        # The squares that will end up being connected
        self.toConnect    = toConnect
        # The pairs that make up the pattern
        self.pairs        = pairs
        # All squares required to be used for the patern
        self.carrier      = {k for pairs in self.pairs for k in pairs}
        # Name of pattern
        self.type         = type1
        # Function to find connected component in original board
        #  used to find out if pattern is completed
        self.findCC       = findCC
        # What is the move that is needed to maintain the pattern
        self.requiredMove = None

    def reply(self, cords : Tuple[int,int]) -> Tuple[int,int] :
        # will return move required to play in response if there is one
        #  and update requiredMove
        if not(cords in self.carrier):
            return
        else:
            for (lp, rp) in self.pairs:
                if cords == lp:
                    self.pairs.remove((lp,rp))
                    self.carrier -= {lp,rp}
                    self.requiredMove = rp
                    return rp
                if cords == rp:
                    self.pairs.remove((lp,rp))
                    self.carrier -= {lp,rp}
                    self.requiredMove = lp
                    return lp
        return
    
    # Perfroms transform on all elements of toConnect, pairs and carrier and requiredMove
    def transform(self, transformer : Callable) -> None:
        self.toConnect = set(map(transformer, self.toConnect))
        self.pairs     = list(map(lambda x : (transformer(x[0]), transformer(x[1])), self.pairs))
        self.carrier      = {k for pairs in self.pairs for k in pairs}
        if self.requiredMove != None:
            self.requiredMove = transformer(self.requiredMove)

    def requiredMoveMade(self, cords:Tuple[int,int]) -> bool:
        # Check if required move has been made
        if self.requiredMove == None:
            return True
 
        if cords == self.requiredMove:
            self.requiredMove = None
            return True
        return False

    def completed(self) -> bool:
        # Is the pattern complete either by: 
        #  - no more pairs left (and no requiredMove)
        #  - all the elements of toConnect are already connected  
        if self.pairs == [] and self.requiredMove == None:
            return True

        first = self.findCC(next(iter(self.toConnect)))
        return all(map(lambda x: self.findCC(x) == first, self.toConnect))

    def show(self):
        print(self.__str__())

    def __str__(self):
        return "{} \\ {} \\ {}".format(self.toConnect, self.pairs, self.type)

    def __repr__(self):
        return self.__str__()

    def __copy__(self) -> 'PairingPattern':
        copp = PairingPattern(self.type, copy(self.toConnect), copy(self.pairs), self.findCC)
        copp.requiredMove = self.requiredMove
        return copp


    def copy(self) -> 'PairingPattern':
        return self.__copy__()

#-------- Datatypes -------------------------------------------------------------

#-------- Destructors -----------------------------------------------------------

#-------- Helper Functions ------------------------------------------------------

#-------- Main Functions --------------------------------------------------------

if __name__ == "__main__":
    print(description)
    a = PairingPattern("Testing", {0, 5 ,6}, [(1,2), (3,4)], lambda x: x)
    a.reply(4)
    print(a.requiredMove)


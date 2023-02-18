# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=True
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True
# cython: profile=True

description = """{-
Module      : HSearch
Description : Defines HSearch Algorithm
Maintainer  : River


#    A B C D E 
#   --------------
#   1 \\ . . . A . \\ 1
#    2 \\ . F . . B \\ 2
#     3 \\ . . X . . \\ 3
#      4 \\ E . . C . \\ 4
#       5 \\ . D . . . \\ 5
#           -------------
#              A B C D E

Axis : 
below is a diagram of all 6 axis on a hex board :
  0 - runs from a to b
  1 - runs from c to d
  2 - runs from e to f
  3 - runs from g to h
  4 - runs from i to j
  5 - runs from k to l

These axis come in pairs definining rotations of x,y axis :
  axis n goes with (n + 1)

#    A B C D E F 
#   ---------------
# 1 \\ . j l c . \\ 1
#  2 \\ a . . . b \\ 2
#   3 \\ k . X . e \\ 3
#    4 \\ h . . . g \\ 4
#     5 \\ . i f d . \\ 5
#         -------------
#           A B C D E F


-}"""

#-------- Imports ---------------------------------------------------------------
# To compile Cython code
import pyximport; 
pyximport.install()
import numpy as np

from alphazero.envs.hex.hexBoard cimport Board
from copy import deepcopy

from typing import List, Tuple, Callable, Set
from alphazero.envs.hex.pattern import Pattern, PairingPattern
import time

#-------- Classes ---------------------------------------------------------------

cdef class HSearchingBoard(Board):
    CHECK_FOR_EDGE_TEMPLATES_INTERNAL = False
    cdef list blueVCs
    cdef list redVCs


    def __init__(self, boardSize : int):
        super().__init__(boardSize)
        self.blueVCs = []
        self.redVCs  = []

    def __str__(self) -> str:
        super().__str__() + 


    def __deepcopy__(self):
        temp  = HSearchingBoard(self.boardSize)
        temp.board = np.copy(self.board)
        temp.connectionBoard = np.copy(self.connectionBoard)
        temp.rank  = np.copy(self.rank)
        temp.blueVCs = deepcopy(blueVCs)
        temp.redVCs = deepcopy(redVCs)
        return temp

    def __getstate__(self):
        bSize, b, connectionb, rank = super().__getstate__()
        return bSize, b, connectionb, rank, blueVCs, redVCs

    def __setstate__(self, state):
        self.__setstate__()

    #zigguratPaternList = [t()]
    def axis(self,i, flip = False):
        if (i == 0):
            return lambda v: (v,0)
        elif (i == 1 ):
            return lambda v: (0,v)
        elif (i == 2):
            return lambda v: (-v,v)
        elif (i == 3):
            return lambda v: (-v,0)
        elif (i == 4):
            return lambda v: (0,v)
        elif (i == 5):
            return lambda v: (v,-v)
        else:
            return 0;

    def offSetBy(self, i, start, offset, flip = False):
        ## Offset start by offset[0] in X direction offset[1] in y direction

        xAxis = self.axis(i)
        yAxis = self.axis((i + 1)% 6)

        s = lambda x,y : (x[0]+y[0], x[1]+y[1])

        if flip:
            offset = (-offset[0] - offset[1], offset[1])

        return (s(s(start,xAxis(offset[0])),yAxis(offset[1])));


    def getBridge(self, cords : tuple[int,int]) -> bool:
        # Adds bridges to virtual connections
        # Symetric WRT rotation (i.e only neeed consider where we have played in x)
        #    A B C D E 
        #   --------------
        #   1 \\ . . . a . \\ 1
        #    2 \\ . f . . b \\ 2
        #     3 \\ . . x . . \\ 3
        #      4 \\ e . . c . \\ 4
        #       5 \\ . d . . . \\ 5
        #           -------------
        #              A B C D E
        added = False

        def bridgePattern(i:int):

            connectingTo = [self.offSetBy(i,cords,(1, -2))]
            pairs = [(self.offSetBy(i,cords,(0, -1)), self.offSetBy(i,cords,(1, -1)))]
            return (connectingTo, pairs)


        for majorAxis in range(0,6):
            c,p = bridgePattern(majorAxis)

            if all([self.board.get(cord) == self.me for cord in c]) and \
               all([((self.board.get(cord) == self.board.EMPTY or 
                      self.board.get(cord) == self.me) and not(self.board.isEdge(cord)))
                    for pairCords in p for cord in pairCords]):
                pp = PairingPattern("Bridge", {cords, *c}, p, self.board.findCC)
                if not pp.completed():
                    added = True
                    self.virtualConnections.append(pp)


        return added
    """
    def getWheel(self, cords : tuple[int,int]) -> bool:
        # Adds wheels to virtual connections
        # Symetric WRT rotation (i.e only neeed consider where we have played in x)

        #    A B C D E 
        #   --------------
        #   1 \\ . . . . . \\ 1
        #    2 \\ . . x . . \\ 2
        #     3 \\ . . . b . \\ 3
        #      4 \\ . a . . . \\ 4
        #       5 \\ . . . . . \\ 5
        #           -------------
        #              A B C D E
        added = False

        def wheelPattern(i:int):

            connectingTo = [self.offSetBy(i,cords,(-1, 2)), self.offSetBy(i,cords,(1, 1))]
            pairs = [(self.offSetBy(i,cords,(0, 1)), self.offSetBy(i,cords,(-1, 1))),
                     (self.offSetBy(i,cords,(1, 0)), self.offSetBy(i,cords,(0, 2)))]
            return (connectingTo, pairs)

        for majorAxis in range(0,6):
            c,p = wheelPattern(majorAxis)

            if all([self.internalState.getTotal(cord) == self.me for cord in c]) and \
               all([((self.internalState.getTotal(cord) == self.internalState.EMPTY or 
                      self.internalState.getTotal(cord) == self.me) and type(cord) != str)
                    for pairCords in p for cord in pairCords]):
                pp = PairingPattern("Wheel", {cords, *c}, p, self.realBoardUnionFind)
                if not pp.completed():                
                    added = True
                    self.virtualConnections.append(pp)

        return added

    def getZiggurat(self, cords : tuple[int,int]) -> bool:
        # Adds ziggurat to virtual connections
        #  most likely edge pattern

        #    A B C D E 
        #   --------------
        #   1 \\             \\ 1
        #    2 \\       x 3   \\ 2
        #     3 \\     1 3 1   \\ 3
        #      4 \\   2 2 4 4   \\ 4
        #       5 \\   a b c     \\ 5
        #           -------------
        #              A B C D E

        # Flippped is : 
        #    A B C D E 
        #   --------------
        #   1 \\             \\ 1
        #    2 \\       3 x   \\ 2
        #     3 \\     1 3 1   \\ 3
        #      4 \\   4 4 2 2   \\ 4
        #       5 \\   c b a     \\ 5
        #           -------------
        #              A B C D E

        added = False

        def zigPatterns(i:int, flip : bool):
            offBy = lambda x : self.offSetBy(i,cords,x,flip)
            ret = []
            # The numbers are standard offsets (i.e having played in x)
            #  Then offi and offj are the offsets for each of x,a,b,c from x to get the relative positions
            tocheck = [(0,0), (0,-3), (1,-3), (2,-3)] if self.CHECK_FOR_EDGE_TEMPLATES_INTERNAL else [(0,0)] 

            for  offi,offj in tocheck:
                connectingTo = [offBy((offi + 0, offj + 0)),
                                offBy((offi + 0, offj + 3)),
                                offBy((offi - 1, offj + 3)),
                                offBy((offi - 2, offj + 3))]

                pairs = [(offBy((offi - 1, offj + 1)), offBy((offi + 1, offj + 1))),
                         (offBy((offi - 1, offj + 2)), offBy((offi - 2, offj + 2))),
                         (offBy((offi + 0, offj + 1)), offBy((offi + 1, offj + 0))),
                         (offBy((offi + 0, offj + 2)), offBy((offi + 1, offj + 2))) ]
                
                ret.append((connectingTo, pairs))
            return ret

        for flip in [False, True]:
            for majorAxis in range(0,6):
                csps = zigPatterns(majorAxis, flip)

                for c,p in csps:

                    if all([self.internalState.getTotal(cord) == self.me for cord in c]) and \
                       all([((self.internalState.getTotal(cord) == self.internalState.EMPTY or 
                              self.internalState.getTotal(cord) == self.me) and type(cord) != str)
                            for pairCords in p for cord in pairCords]):
                        pp = PairingPattern("Zigugurat", {cords, *c}, p, self.realBoardUnionFind)
                        if not pp.completed():                
                            added = True
                            self.virtualConnections.append(pp)
        return added

    def getIII1b(self, cords : tuple[int,int]) -> bool:
        # Adds III1b to virtual connections
        #  most likely edge pattern

        #      A B C D E 
        #      --------------
        #   1 \\             \\ 1
        #    2 \\     3 x 4   \\ 2
        #     3 \\   1 3 4 1   \\ 3
        #      4 \\ 2 2   5 5   \\ 4
        #       5 \\ d a b c     \\ 5
        #           -------------
        #              A B C D E
    
        added = False

        def III1bPatterns(i:int):
            offBy = lambda x : self.offSetBy(i,cords,x)
            ret = []
            # The numbers are standard offsets (i.e having played in x)
            #  Then offi and offj are the offsets for each of x,a,b,c from x to get the relative positions
            tocheck = [(0,0), (0,-3), (1,-3), (2,-3), (3,-3)] if self.CHECK_FOR_EDGE_TEMPLATES_INTERNAL else [(0,0)] 

            for  offi,offj in tocheck:
                connectingTo = [offBy((offi + 0, offj + 0)),
                                offBy((offi + 0, offj + 3)),
                                offBy((offi - 1, offj + 3)),
                                offBy((offi - 2, offj + 3)),
                                offBy((offi - 3, offj + 3))]

                pairs = [(offBy((offi - 2, offj + 1)), offBy((offi + 1, offj + 1))),
                         (offBy((offi - 2, offj + 2)), offBy((offi - 3, offj + 2))),
                         (offBy((offi - 1, offj + 1)), offBy((offi - 1, offj + 0))),
                         (offBy((offi + 1, offj + 0)), offBy((offi + 0, offj + 1))),
                         (offBy((offi + 0, offj + 2)), offBy((offi + 1, offj + 2)))]
                
                ret.append((connectingTo, pairs))
            return ret

        for majorAxis in range(0,6):
            csps = III1bPatterns(majorAxis)

            for c,p in csps:
                if all([self.internalState.getTotal(cord) == self.me for cord in c]) and \
                   all([((self.internalState.getTotal(cord) == self.internalState.EMPTY or 
                          self.internalState.getTotal(cord) == self.me) and type(cord) != str)
                        for pairCords in p for cord in pairCords]):
                    pp = PairingPattern("III3b", {*c}, p, self.realBoardUnionFind)
                    if not pp.completed():                
                        added = True
                        self.virtualConnections.append(pp)
        return added

    def getIV1a(self, cords : tuple[int,int]) -> bool:
        # Adds III1b to virtual connections
        #  most likely edge pattern

        #       A B C D E 
        #       --------------
        #    1 \\         5 x 6   \\ 1
        #     2 \\     4 1 5 6 1 7 \\ 2
        #      3 \\   2 4 2   3 7 3 \\ 3
        #       4 \\ 8 8 9 9 α α β β \\ 4
        #        5 \\ a b c d e f g   \\ 5
        #           -------------
        #              A B C D E

        added = False

        def IV1aPatterns(i:int):
            offBy = lambda x : self.offSetBy(i,cords,x)
            ret = []
            # The numbers are standard offsets (i.e having played in x)
            #  Then offi and offj are the offsets for each of x,a,b,c from x to get the relative positions
            tocheck = [(0,0), (0,-4), (-1,-4), (1,-4), (2,-4), (3,-4), (4,-4), (5,-4)] if self.CHECK_FOR_EDGE_TEMPLATES_INTERNAL else [(0,0)] 

            for  offi,offj in tocheck:
                connectingTo = [offBy((offi + 0, offj + 0)),
                                offBy((offi + 0, offj + 4)),
                                offBy((offi + 1, offj + 4)),
                                offBy((offi - 1, offj + 4)),
                                offBy((offi - 2, offj + 4)),
                                offBy((offi - 3, offj + 4)),
                                offBy((offi - 4, offj + 4)),
                                offBy((offi - 5, offj + 4))]

                pairs = [(offBy((offi - 2, offj + 1)), offBy((offi + 1, offj + 1))), #1
                         (offBy((offi - 2, offj + 2)), offBy((offi - 4, offj + 2))), #2
                         (offBy((offi + 0, offj + 2)), offBy((offi + 2, offj + 2))), #3
                         (offBy((offi - 3, offj + 1)), offBy((offi - 3, offj + 2))), #4
                         (offBy((offi - 1, offj + 0)), offBy((offi - 1, offj + 1))), #5
                         (offBy((offi + 1, offj + 0)), offBy((offi + 0, offj + 1))), #6
                         (offBy((offi + 2, offj + 1)), offBy((offi + 1, offj + 2))), #7
                         (offBy((offi - 5, offj + 3)), offBy((offi - 4, offj + 3))), #8
                         (offBy((offi - 3, offj + 3)), offBy((offi - 2, offj + 3))), #9
                         (offBy((offi - 1, offj + 3)), offBy((offi + 0, offj + 3))), #α
                         (offBy((offi + 1, offj + 3)), offBy((offi + 2, offj + 3)))] #β

                
                ret.append((connectingTo, pairs))
            return ret

        for majorAxis in range(0,6):
            csps = IV1aPatterns(majorAxis)

            for c,p in csps:
                if all([self.internalState.getTotal(cord) == self.me for cord in c]) and \
                   all([((self.internalState.getTotal(cord) == self.internalState.EMPTY or 
                          self.internalState.getTotal(cord) == self.me) and type(cord) != str)
                        for pairCords in p for cord in pairCords]):
                    pp = PairingPattern("IV1a", {*c}, p, self.realBoardUnionFind)
                    if not pp.completed():                
                        added = True
                        self.virtualConnections.append(pp)

        return added
    """
    def getNewVCs(self,cords : tuple[int,int]) -> bool:
        aBridge    = self.getBridge(cords)
        #aWheel     = self.getWheel(cords)
        #aZigugurat = self.getZiggurat(cords)
        #aIIIb      = self.getIII1b(cords)
        #aIVa      = self.getIV1a(cords)

        return any([aBridge])#, aWheel, aZigugurat, aIIIb, aIVa])



    def updateWith(self, cords : tuple[int,int], player : int)-> list[(set[(int,int)], (int,int))]:
        ret = []
        if player == self.me:
            self.getNewVCs(cords)

            for pat in self.virtualConnections:
                if not(pat.requiredMoveMade(cords)):
                    self.virtualConnections.remove(pat)

        else:
            for pat in self.virtualConnections:
                if pat.completed():
                    self.virtualConnections.remove(pat)
            
            for pat in self.virtualConnections:
                rep = pat.reply(cords)
                if rep != None:
                    ret.append((pat.toConnect, rep))

        return ret

    # A sufficient number of extra pieces that need to be placed for self.me to win 
    #  with 100% confidence if tempCords was played (if it is valid)
    #  i.e to create a virtual connection between self.me's 2 halves of the board
    # Bredth first search
    def getBridgeDistance(self,tempCords = (ERROR,ERROR)):
        # init visted for search
        # Represents lowest cost for a node that has been seen so far (but not necessarily 
        #   expanded yet)
        if tempCords[0] != self.internalState.ERROR:
            vc = copy(self.virtualConnections)
            self.internalState.update(tempCords, self.me)
            self.getNewVCs(tempCords)

        visited = {(i,j):np.inf for i in range(0, self.internalState.boardSize) for j in range(0, self.internalState.boardSize)}
        for i in ["BlueLeft", "BlueRight", "RedUpper", "RedLower"]:
            visited[i] = np.inf

        fromSide = None
        toSide   = None
        if self.me == self.internalState.BLUE_PLAYER:
            fromSide = "BlueLeft"
            toSide   = "BlueRight"
        else:
            fromSide = "RedUpper"
            toSide   = "RedLower"

        visited[fromSide] = 0

        toExplore = [(fromSide,0)]

        while toExplore != []:
            #print(toExplore)
            nextToExpand, cost = toExplore.pop(0)
            if cost > visited.get(nextToExpand):
                continue

            for pat in self.virtualConnections:
                if nextToExpand in pat.toConnect:
                    # Then add in all virtually connected bits
                    for toAdd in pat.toConnect:
                        if toAdd == nextToExpand:
                            continue;
                        if cost < visited[toAdd]:
                            toExplore.append((toAdd, cost))
                            visited[toAdd] = cost

            for neighbour in self.internalState.getNeighboursTotal(nextToExpand):
                if self.internalState.getTotal(neighbour) == self.them:
                    continue

                newCost = cost + (1 if (self.internalState.getTotal(nextToExpand) != self.me) else 0)
                if newCost < visited[neighbour]:
                    toExplore.append((neighbour, newCost))
                    visited[neighbour] = newCost

        if tempCords[0] != self.internalState.ERROR:
            self.internalState.update(tempCords, self.internalState.EMPTY)
            self.virtualConnections = vc

        return visited[toSide]







#-------- Datatypes -------------------------------------------------------------

#-------- Destructors -----------------------------------------------------------

#-------- Helper Functions ------------------------------------------------------

#-------- Main Functions --------------------------------------------------------


if __name__ == "__main__":
    b = Board(13)
    hSearcher = HSearcher(1, b)

    for i in [(0,0), (1,1)]:
        b.play(i, 1)
        hSearcher.updateWith(i,1)

    hSearcher.updateWith((0,1), 0)
    for i in hSearcher.virtualConnections:
        print(i)
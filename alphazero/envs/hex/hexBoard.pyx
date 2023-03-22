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
Module      : Board
Description : Defines the board and all access that needs to be done on it; it 
                enforces the physical invariants of the board (i.e. can't have
                2 pieces on same place on board)

# This is a hex board the colums are diagonal (as it's made of hexagons) :
#  - The cords work in x and y axis with upper right being (0,0) so for example
#    - 1 is in position (2,0)
#    - 2 is in position (6,1)
#  - There are marked edges on the board where R represents being owned by Red, B represents blue and 
#      G represents it should be treated as both
#   
#   A B C D E F G H I J K
#   --------------------------
#     G R R R R R R R R R R R G
# 1 \\ B . . 1 . . . . . . . . B \\ 1
#  2 \\ B . . . . . . 2 . . . . B \\ 2
#   3 \\ B . . . . . . . . . . . B \\ 3
#    4 \\ B . . . . . . . . . . . B \\ 4
#     5 \\ B . . . . . . . . . . . B \\ 5
#      6 \\ B . . . . . . . . . . . B \\ 6
#       7 \\ B . . . . . . . . . . . B \\ 7
#        8 \\ B . . . . . . . . . . . B \\ 8
#         9 \\ B . . . . . . . . . . . B \\ 9
#          10\\ B . . . . . . . . . . . B \\10
#           11\\ B . . . . . . . . . . . B \\11
#                 G R R R R R R R R R R R G
#              --------------------------
#                 A B C D E F G H I J K


Maintainer  : River
-}"""
#
#-------- Imports ---------------------------------------------------------------

#-----------------------
from typing import List, Tuple, Any
import numpy as np
from copy import deepcopy
from alphazero.envs.hex.pattern import Pattern, PairingPattern
#-------- Classes ---------------------------------------------------------------

EMPTY = 0
BLUE_PLAYER = 1
RED_PLAYER = 2
GREEN_PLAYER = 3
ERROR = -10
SWAP  = -9
SWAP_MOVE = False
CHECK_FOR_EDGE_TEMPLATES_INTERNAL = False
            # Brdige, Wheel, Ziggurat, III1b, IV1a, 
vcsToCheck = [True,  False, False,    False, False]

SHOW_VCs = any(vcsToCheck)


cdef class Board():
    # Players
    cdef public int EMPTY
    cdef public int BLUE_PLAYER 
    cdef public int RED_PLAYER
    # Green simply represents a square is held by both red and blue
    cdef public int GREEN_PLAYER
    cdef public int ERROR
    cdef public int SWAP
    cdef public int boardSize
    cdef public int[:] board
    cdef public int[:] connectionBoard
    cdef public int[:] rank
    cdef public tuple lastMove

    cdef public list blueVCs
    cdef public list redVCs
    #cdef tuple lastVCsAdded

    def initconsts(self, int boardSize):
        self.EMPTY        = EMPTY
        self.BLUE_PLAYER  = BLUE_PLAYER
        self.RED_PLAYER   = RED_PLAYER
        self.GREEN_PLAYER = GREEN_PLAYER
        self.ERROR        = ERROR
        self.SWAP         = SWAP

    def __init__(self, int boardSize):
        self.initconsts(boardSize)
        # Details of board
        self.boardSize = boardSize
        cdef int length = (boardSize+2) * (boardSize + 2)
        # Board representing board (+ edges) 
        self.board = np.zeros(length, dtype=np.intc)
        self.lastMove = ((self.ERROR, self.ERROR), self.ERROR)

        # This is replacing union find as will be easier to before symetries on
        #  Given a cell will return another cell which is in the same connected component
        #    The "leader" 
        self.connectionBoard = np.arange(length, dtype=np.intc)
        # MAY REMOVE RANK??
        # Rank is the rank in the UFDS 
        #   provides an upper bound on the height of the tree formed by UFDS
        self.rank            = np.zeros(length, dtype=np.intc)

        # Add RedUpper
        self.connectionBoard[1:boardSize+1] = 1
        self.rank           [1] = 1
        self.board          [1:boardSize+1] = RED_PLAYER
        # Add RedLower
        self.connectionBoard[length-boardSize-1:length-1] = length-boardSize-1
        self.rank           [length-boardSize-1] = 1
        self.board          [length-boardSize-1:length-1] = RED_PLAYER
        # Add BlueLeft
        self.connectionBoard[boardSize+2:length - boardSize-2:boardSize+2] = boardSize+2
        self.rank           [boardSize+2] = 1
        self.board          [boardSize+2:length - boardSize-2:boardSize+2] = BLUE_PLAYER
        # Add BlueRight
        self.connectionBoard[2*boardSize+3:length - 1:boardSize+2] = 2*boardSize+3
        self.rank           [2*boardSize+3] = 1
        self.board          [2*boardSize+3:length - 1:boardSize+2] = BLUE_PLAYER

        # Add Corners
        self.board[0], self.board[boardSize+1], self.board[-1], self.board[-(boardSize+2)] = [GREEN_PLAYER]*4
        
        # Initialise VCs
        self.blueVCs = []
        self.redVCs  = []

    # Returns string representing the board
    def __str__(self) -> str:
        ret = ""
        switch = {
            EMPTY          : ".",
            BLUE_PLAYER    : "B",
            RED_PLAYER     : "R"
        }
        ret += "   "
        for x in range(0,self.boardSize): 
            ret += chr(65+x) + " "
        ret += "\n"

        for y in range(0,self.boardSize):
            ret += " "*(y+1) + str(y+1) + " "*(y<9) + " "
            for x in range(0,self.boardSize):
                at = self.get((x, y))
                p = switch.get(at) if (at in switch) else at
                ret += p + " " 
            ret += "\n"

        if SHOW_VCs:
            ret += "\n  Blue's VC's :\n"
            for i in self.blueVCs:
                ret += "    {}\n".format(i)

            ret += "\n  Red's VC's :\n"
            for i in self.redVCs:
                ret += "    {}\n".format(i)

        return ret
        
    # Copies self
    def __deepcopy__(self) -> 'Board':
        temp                 = Board(self.boardSize)
        temp.board           = np.copy(self.board)
        temp.connectionBoard = np.copy(self.connectionBoard)
        temp.rank            = np.copy(self.rank)
        temp.blueVCs         = deepcopy(self.blueVCs)
        temp.redVCs          = deepcopy(self.redVCs)

        return temp

    def __getstate__(self):
        return self.boardSize, np.asarray(self.board), np.asarray(self.connectionBoard), np.asarray(self.rank), self.blueVCs, self.redVCs

    def __setstate__(self, state):
        self.boardSize, board, connectionBoard, rank, blueVCs, redVCs = state
        self.initconsts(self.boardSize)
        self.board           = np.asarray(board)
        self.connectionBoard = np.asarray(connectionBoard)
        self.rank            = np.asarray(rank)
        self.blueVCs         = blueVCs
        self.redVCs          = redVCs

    # Equivalent to __copy__
    def deepcopy(self) -> 'Board':
        return self.__deepcopy__()

    # _.show() equivalent to print(_)
    def show(self) -> None:
        print(self.__str__())

    # Defines the axis the board can be looked at from
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
            return lambda v: (0,-v)
        elif (i == 5):
            return lambda v: (v,-v)
        else:
            return 0;

    ## Offset start by offset[0] in x direction offset[1] in y direction
    #  flip if required
    def offSetBy(self, axisDir, start, offset, flip = False):

        xAxis = self.axis(axisDir)
        yAxis = self.axis((axisDir + 1)% 6)

        s = lambda x,y : (x[0]+y[0], x[1]+y[1])

        if flip:
            offset = (-offset[0] - offset[1], offset[1])

        return (s(s(start,xAxis(offset[0])),yAxis(offset[1])));

    # Symetries of the board
    def rotate(self) -> None:
        # Symmetry 1 : A rotation of pi about x (center of board)
        #   - note board could be even sized
        #      A B C D E                    A B C D E             
        #     --------------               --------------           
        #   1 \\ 1 2 3 4 5 \\ 1          1 \\ . . . . . \\ 1        
        #    2 \\ 6 . . . . \\ 2          2 \\ . . . . . \\ 2      
        #     3 \\ . . x . . \\ 3    ->    3 \\ . . x . . \\ 3  
        #      4 \\ . . . . . \\ 4          4 \\ . . . . 6 \\ 4     
        #       5 \\ . . . . . \\ 5          5 \\ 5 4 3 2 1 \\ 5     
        #           -------------                -------------       
        #              A B C D E                    A B C D E    
        size = (self.boardSize+2) 

        # maps a possition on original board to flipped board
        def rotationCorrection(x: int, bSize:int) -> int:
            return bSize*bSize-1-x
        def pairCorrection(x:Tuple[int,int]) -> Tuple[int,int]:
            return self.intToCords(rotationCorrection(self.cordsToInt(x), size))

        self.board             = np.flipud(np.fliplr(np.resize(self.board,(size,size)))).ravel()
        self.rank              = np.flipud(np.fliplr(np.resize(self.rank, (size,size)))).ravel()
        rotatedConnectionBoard = np.flipud(np.fliplr(np.resize(self.connectionBoard, (size,size)))).ravel()
        self.connectionBoard   = rotationCorrection(rotatedConnectionBoard, size)
        
        for blueVC in self.blueVCs:
            blueVC.transform(pairCorrection)

        for redVC in self.redVCs:
            redVC.transform(pairCorrection)
 

    def flipInXPlusY(self, colourCorrect = True) -> None:
        # A reflection in x + y = BOARD_SIZE and a colour swap
        #      A B C D E 
        #     --------------
        #   1 \\ . . . . / \\ 1
        #    2 \\ . . . / . \\ 2
        #     3 \\ . . / . . \\ 3
        #      4 \\ . / . . . \\ 4
        #       5 \\ / . . . . \\ 5
        #           -------------
        #              A B C D E
        cdef int size = (self.boardSize+2) 

        colourFlipper = lambda x : 2*(x==1) + 1*(x==2) + x*(x!=1 and x!=2)

        boardWithoutCorrection = np.flipud(np.fliplr(np.transpose(np.resize(np.asarray(self.board), (size,size))))).ravel()
        self.board             = np.vectorize(colourFlipper)(boardWithoutCorrection).astype(np.intc) if colourCorrect else boardWithoutCorrection.astype(np.intc)
        self.rank              = np.flipud(np.fliplr(np.transpose(np.resize(np.asarray(self.rank), (size,size))))).ravel()
        flippedConnectionBoard = np.flipud(np.fliplr(np.transpose(np.resize(np.asarray(self.connectionBoard), (size,size))))).ravel()
        self.connectionBoard   = np.vectorize(self.flipInXPlusYCorrection)(flippedConnectionBoard).astype(np.intc)

        for blueVC in self.blueVCs:
            blueVC.transform(self.pairflipInXPlusYCorrection)

        for redVC in self.redVCs:
            redVC.transform(self.pairflipInXPlusYCorrection)

        self.blueVCs,self.redVCs = (self.redVCs,self.blueVCs)
 

    def pairflipInXPlusYCorrection(self, x: Tuple[int,int]) -> Tuple[int,int]:
        return (self.boardSize - 1 - x[1], self.boardSize - 1 - x[0])

    # maps a possition on original board to flipped board
    def flipInXPlusYCorrection(self, int x) -> int:
        return self.cordsToInt(self.pairflipInXPlusYCorrection(self.intToCords(x)))

    # Given a cell returns the leader of it's connected component (the leader has parent )
    # Input  : The cell's cords as an int (representing where it is on !!!board!!! [VERY IMPORTANT])
    # Output : The cell's leaders cords as an int 
    def findCCInt(self, int cordsInt) -> int:
        parent = self.connectionBoard[cordsInt]

        if (parent == cordsInt):
            return cordsInt

        ret = self.findCCInt(parent)
        if (parent != ret):
            self.connectionBoard[cordsInt] = ret
        return ret

    # Same as above but with the x,y representation
    def findCC(self, cords: Tuple[int, int]) -> Tuple[int, int]:
        return self.intToCords(self.findCCInt(self.cordsToInt(cords)))

    # Given some x,y cordinates returns the int to represent their location on
    #  Board -> will translate the edges correctly, i.e. for acceptable _
    #     (_, -1)          -> RedTop
    #     (_, boardSize) -> RedBottom
    #     (-1, _)          -> BlueLeft
    #     (boardSize, _) -> BlueRight
    cpdef public int cordsToInt(self, (int,int) cords):
        cdef int x = cords[0]
        cdef int y = cords[1]
        return (self.boardSize + 2) * (y+1) + (x+1)

    # inverse of cordsToInt
    cpdef public tuple intToCords(self, int cordsInt):
        cdef int x = cordsInt%(self.boardSize + 2) - 1
        cdef int y = cordsInt//(self.boardSize + 2) - 1
        return (x,y)

    # Returns the element at (x,y)
    cpdef public int get(self, (int,int) cords):
        cdef int i = self.cordsToInt(cords)
        cdef int length = len(self.board)
        if i < length and i >= 0:
            return self.board[i];
        else:
            return self.ERROR

    # Updates board[n] to be l
    cdef void update(self, int cordsInt, int l):
        self.board[cordsInt] = l

    # Returns if board[i][j] == EMPTY
    def canPlay(self, cords: Tuple[int, int]) -> bool:
        return self.get(cords) == self.EMPTY;

    # Returns if board at i j is the edge
    def isEdge(self, cords:Tuple[int,int]) -> bool:
        return cords[0] == -1 or cords[1] == -1 or cords[0] == self.boardSize or cords[1] == self.boardSize

    # Swaps red and blue's positions; implemented in the littlegolem style
    def swap(self):
        self.rotate()
        self.flipInXPlusY()

    # Plays a piece in i,j and updates turn and the ufds (i.e self.connectionBoard)
    # PRE : isValid(i,j)
    #    && canPlay(i,j)
    def play(self, cords: Tuple[int, int], player: int):
        if cords == (self.SWAP,self.SWAP):
            self.swap()
            return;

        assert self.canPlay(cords), "failing cords : {} it is {}".format(cords, self.get(cords))
        cdef int cordsInt = self.cordsToInt(cords)
        #print(cordsInt)
        self.lastMove = (cords, player)
        # Play Move
        self.update(cordsInt, player)

        cdef int myLeader
        cdef int theirLeader

        # Union by rank where appropriate (where neightbour is same colour)
        for c in self.getNeighbours(cords):
            if self.get(c) == player:
                myLeader    = self.findCCInt(cordsInt)
                theirLeader = self.findCCInt(self.cordsToInt(c))
                
                if myLeader == theirLeader:
                    continue

                if self.rank[myLeader]>self.rank[theirLeader]:
                    self.connectionBoard[theirLeader] = myLeader
                elif self.rank[theirLeader]>self.rank[myLeader]:
                    self.connectionBoard[myLeader] = theirLeader
                else:
                    self.connectionBoard[myLeader] = theirLeader
                    self.rank[theirLeader] += 1

        self.updateWith(cords, player)

    def getNewVCs(self, cords : Tuple[int,int], player : int) -> None:
        myVCs = self.blueVCs if (player == BLUE_PLAYER) else self.redVCs
        ret   = []

        funcList = [self.getBridge, self.getWheel, self.getZiggurat, self.getIII1b, self.getIV1a]
        
        for i in range(0, len(funcList)):
            if vcsToCheck[i]:
                ret.extend(funcList[i](cords, player))

        myVCs.extend(ret)
        #self.lastVCsAdded = (self.lastVCsAdded[1], ret)
        return None

    def updateWith(self, cords : Tuple[int,int], player : int) -> None:
        ret      = []
        theirVCs = self.redVCs if (player == BLUE_PLAYER) else self.blueVCs

        # Update Your VCs
        self.getNewVCs(cords, player)
        # If haven't made a required move or have completed a pattern remove it
        if (player == BLUE_PLAYER):
            self.blueVCs = [pat for pat in self.blueVCs if pat.requiredMoveMade(cords) and not(pat.completed())]
        else:
            self.redVCs  = [pat for pat in self.redVCs if pat.requiredMoveMade(cords) and not(pat.completed())]

        # Generate replies in oponents pattersn
        for pat in theirVCs:
            pat.reply(cords)

        return None;

    # Adds bridges to virtual connections
    # Symetric WRT rotation (i.e only neeed consider where we have played in x)
    def getBridge(self, cords : Tuple[int,int], me : int) -> List[Pattern]:
        #    A B C D E 
        #   --------------
        #   1 \\ . . . a . \\ 1
        #    2 \\ . f . . b \\ 2
        #     3 \\ . . x . . \\ 3
        #      4 \\ e . . c . \\ 4
        #       5 \\ . d . . . \\ 5
        #           -------------
        #              A B C D E
        ret = []
        for majorAxis in range(0,6):
            connectingTo = self.offSetBy(majorAxis,cords,(1, -2))
            pairs = [(self.offSetBy(majorAxis,cords,(0, -1)), self.offSetBy(majorAxis,cords,(1, -1)))]

            # Check that :
            #   - What we are connecting to is owned by us
            #   - the carrier of the pattern is empty or owned by us
            #   - the carrier doesn't include an edge (get's rid of edge cases)
            if self.get(connectingTo) == me and \
               all([(self.get(cord) == self.EMPTY and not(self.isEdge(cord)))
                    for pairCords in pairs for cord in pairCords]):


                pp = PairingPattern("Bridge", {cords, connectingTo}, pairs, self.findCC)
                if not pp.completed():
                    ret.append(pp)


        return ret
    
    # Adds wheels to virtual connections
    # Symetric WRT rotation (i.e only neeed consider where we have played in x)
    def getWheel(self, cords : tuple[int,int], me : int) -> List[Pattern]:
        #    A B C D E 
        #   --------------
        #   1 \\ . . . . . \\ 1
        #    2 \\ . . x . . \\ 2
        #     3 \\ . . . b . \\ 3
        #      4 \\ . a . . . \\ 4
        #       5 \\ . . . . . \\ 5
        #           -------------
        #              A B C D E
        ofs = self.offSetBy
        ret = []
        for majorAxis in range(0,6):
            connectingTo = [ofs(majorAxis,cords,(-1, 2)), ofs(majorAxis,cords,(1, 1))]
            pairs = [(ofs(majorAxis,cords,(0, 1)), ofs(majorAxis,cords,(-1, 1))),
                     (ofs(majorAxis,cords,(1, 0)), ofs(majorAxis,cords,( 0, 2)))]
            
            if all([self.get(cord) == me for cord in connectingTo]) and \
               all([(self.get(cord) == self.EMPTY and not(self.isEdge(cord)))
                    for pairCords in pairs for cord in pairCords]):
                pp = PairingPattern("Wheel", {cords, *connectingTo}, pairs, self.findCC)
                if not pp.completed():                
                    ret.append(pp)

        return ret

    # Adds ziggurat to virtual connections
    #  most likely edge pattern
    def getZiggurat(self, cords : tuple[int,int], me : int) -> List[Pattern]:
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

        ret = []

        # "Positions" of our placed stone in the general shape
        tocheck = [(0,0), (0,-3), (1,-3), (2,-3)] if CHECK_FOR_EDGE_TEMPLATES_INTERNAL else [(0,0)] 

        for flip in [False, True]:
            for majorAxis in range(0,6):
                ofs  = lambda x : self.offSetBy(majorAxis,cords,x,flip)
                for  offi,offj in tocheck:
                    connectingTo = [ofs((offi + 0, offj + 0)),
                                    ofs((offi + 0, offj + 3)),
                                    ofs((offi - 1, offj + 3)),
                                    ofs((offi - 2, offj + 3))]

                    unPreProcessedpairs = [(ofs((offi - 1, offj + 1)), ofs((offi + 1, offj + 1))),
                                           (ofs((offi - 1, offj + 2)), ofs((offi - 2, offj + 2))),
                                           (ofs((offi + 0, offj + 1)), ofs((offi + 1, offj + 0))),
                                           (ofs((offi + 0, offj + 2)), ofs((offi + 1, offj + 2))) ]
                

                    pairs = [p for p in unPreProcessedpairs if (self.get(p[0]) != me and self.get(p[1]) != me)]

                    if all([self.get(cord) == me for cord in connectingTo]) and \
                       all([(self.get(cord) == self.EMPTY and not(self.isEdge(cord)))
                            for pairCords in pairs for cord in pairCords]):
                        pp = PairingPattern("Zigugurat", {*connectingTo}, pairs, self.findCC)
                        if not pp.completed():
                            #self.show()
                            #print(unPreProcessedpairs)
                            #print([(self.get(p[0]) != me and self.get(p[1]) != me) for p in unPreProcessedpairs])
                            ret.append(pp)
        return ret

    # Adds III1b to virtual connections
    #  most likely edge pattern
    def getIII1b(self, cords : tuple[int,int], me : int) -> List[Pattern]:
        #      A B C D E 
        #      --------------
        #   1 \\             \\ 1
        #    2 \\     3 x 4   \\ 2
        #     3 \\   1 3 4 1   \\ 3
        #      4 \\ 2 2   5 5   \\ 4
        #       5 \\ d a b c     \\ 5
        #           -------------
        #              A B C D E
    
        ret = []
        # "Positions" of our placed stone in the general shape
        tocheck = [(0,0), (0,-3), (1,-3), (2,-3), (3,-3)] if CHECK_FOR_EDGE_TEMPLATES_INTERNAL else [(0,0)] 

        for majorAxis in range(0,6):
            offBy = lambda x : self.offSetBy(majorAxis,cords,x)
            for  offi,offj in tocheck:
                connectingTo = [offBy((offi + 0, offj + 0)),
                                offBy((offi + 0, offj + 3)),
                                offBy((offi - 1, offj + 3)),
                                offBy((offi - 2, offj + 3)),
                                offBy((offi - 3, offj + 3))]

                unPreProcessedpairs = [(offBy((offi - 2, offj + 1)), offBy((offi + 1, offj + 1))),
                                       (offBy((offi - 2, offj + 2)), offBy((offi - 3, offj + 2))),
                                       (offBy((offi - 1, offj + 1)), offBy((offi - 1, offj + 0))),
                                       (offBy((offi + 1, offj + 0)), offBy((offi + 0, offj + 1))),
                                       (offBy((offi + 0, offj + 2)), offBy((offi + 1, offj + 2)))]
                
                pairs = [p for p in unPreProcessedpairs if (self.get(p[0]) != me and self.get(p[1]) != me)]


                if all([self.get(cord) == me for cord in connectingTo]) and \
                   all([(self.get(cord) == self.EMPTY and not(self.isEdge(cord)))
                        for pairCords in pairs for cord in pairCords]):
                    pp = PairingPattern("III3b", {*connectingTo}, pairs, self.findCC)
                    if not pp.completed():
                        ret.append(pp)
        return ret

    # Adds III1b to virtual connections
    #  most likely edge pattern
    def getIV1a(self, cords : tuple[int,int], me : int) -> List[Pattern]:

        #       A B C D E 
        #       --------------
        #    1 \\         5 x 6   \\ 1
        #     2 \\     4 1 5 6 1 7 \\ 2
        #      3 \\   2 4 2   3 7 3 \\ 3
        #       4 \\ 8 8 9 9 α α β β \\ 4
        #        5 \\ a b c d e f g   \\ 5
        #           -------------
        #              A B C D E

        ret = []
        # The numbers are standard offsets (i.e having played in x)
        #  Then offi and offj are the offsets for each of x,a,b,c from x to get the relative positions
        tocheck = [(0,0), (0,-4), (-1,-4), (1,-4), (2,-4), (3,-4), (4,-4), (5,-4)] if self.CHECK_FOR_EDGE_TEMPLATES_INTERNAL else [(0,0)] 

        for majorAxis in range(0,6):

            offBy = lambda x : self.offSetBy(majorAxis,cords,x)
            for  offi,offj in tocheck:

                connectingTo = [offBy((offi + 0, offj + 0)),
                                offBy((offi + 0, offj + 4)),
                                offBy((offi + 1, offj + 4)),
                                offBy((offi - 1, offj + 4)),
                                offBy((offi - 2, offj + 4)),
                                offBy((offi - 3, offj + 4)),
                                offBy((offi - 4, offj + 4)),
                                offBy((offi - 5, offj + 4))]

                unPreProcessedpairs = [(offBy((offi - 2, offj + 1)), offBy((offi + 1, offj + 1))), #1
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

                pairs = [p for p in unPreProcessedpairs if (self.get(p[0]) != me and self.get(p[1]) != me)]

                if all([self.get(cord) == me for cord in connectingTo]) and \
                   all([(self.get(cord) == self.EMPTY and not(self.isEdge(cord)))
                        for pairCords in pairs for cord in pairCords]):
                    pp = PairingPattern("IV1a", {*connectingTo}, pairs, self.realBoardUnionFind)
                    if not pp.completed():
                        ret.append(pp)

        return ret

    # Returns an array over all positions with ones where it possible to play 
    #  with a 1 and not with a zero
    #  Does not return weather swap possible
    def getPossibleMoves(self):
        size = self.boardSize + 2
        playableExtended = np.reshape(np.where(np.asarray(self.board) == 0, 1, 0), (size,size))
        playableMinus = np.delete(np.delete(playableExtended,[0,-1],1),[0,-1],0)
        playable = playableMinus.ravel().astype(np.int8)

        #print(np.asarray(self.playable))

        return np.asarray(playable)


    # Returns all neighbouring hexes to i,j
    # PRE : isValid(i,j)
    def getNeighbours(self, cords: Tuple[int, int]):
        i,j = cords
        neighbours = []
        for row in range(-1, 2):
            for col in range(-1, 2):
                if row != col:
                    if self.isValid((i + row, j + col)):
                        neighbours.append((i + row, j + col))

        return neighbours

    # Returns if i,j is a valid location for 
    def isValid(self, cords: Tuple[int, int]) -> bool:
        try:
            return all(-1 <= _ < self.boardSize+1 for _ in cords) 
        except:
            return False

#-------- Datatypes -------------------------------------------------------------

#-------- Destructors -----------------------------------------------------------

#-------- Helper Functions ------------------------------------------------------

#-------- Main Functions --------------------------------------------------------

if __name__ == "__main__":
    print(description)
    s = 7
    a = Board(s-2)
    for m,p in [((1,2),2), ((0,2),1), ((2,0),2)]:
        a.play(m,p)
    
    print(a)
    print(a.getNeighbours((0,0)))

# bad code by thomas lam 2022
# pls dont steal thank
#
# TODO:
# - there is a BUG for n = 3,5 mod 8 concerning parity.  will not solve if it is odd permutation.
# - let user hit autosolve key
# - progress bar
# - more color schemes

import pygame
import sys
import time
import random
import math
import copy
from pygame.locals import *

pygame.init()

HEIGHT = 820 # 800 x 1000 good
WIDTH = 1000
screen = pygame.display.set_mode((WIDTH,HEIGHT))
board_center = (WIDTH/2,HEIGHT*0.44)

pygame.display.set_caption('NRP Player')

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

pygame.font.init()
my_font = pygame.font.SysFont('arial', 30)

pygame.key.set_repeat(150)

### CONFIG ###

SQSIDE = 50 # 50 is good
LINEWIDTH = 2

N = 11
M = N+1

SCRAMBLE = False
NUM_SCRAMBLE_MOVES = 500

AUTO_SOLVE = False
AUTO_MOVE_TIME = 100

SHOW_COMPUTATION = True

NRP_rect = Rect(board_center[0]-int(1.5*(SQSIDE*M/2+LINEWIDTH)), 
                board_center[1]-int(1.5*(SQSIDE*N/2+LINEWIDTH)), 
                2*int(1.5*(SQSIDE*M/2+LINEWIDTH)),
                2*int(1.5*(SQSIDE*N/2+LINEWIDTH)))

info_rect = Rect(0, board_center[1]+int(1.5*(SQSIDE*N/2+LINEWIDTH)),WIDTH,HEIGHT)

def redCycle(i,j, M, N):
    if i == 0 or i == M-1 or j == N-1:
        return (255, 100, 100)
    else:
        return WHITE
        
def firstColumn(i,j,M,N):
    if N%2 == 0:
        if (i,j) in [(0,0),(0,1)]:
            return (0, 255, 0)
    else:
        if (i,j) in [(0,0),(0,2)]:
            return (0, 255, 0)
    
    if i == 0:
        return (0, 0, 255)
    else:
        return WHITE
        
def duskGradient(i,j, M, N):
    return (int((1-i/M)*155+100), 100, int(j/M*155+100))

getCellColor = duskGradient

def writeText(screen, text, x, y):
    text_screen = my_font.render(text, False, (0, 0, 0))
    text_rect = text_screen.get_rect()
    screen.blit(text_screen, (x-text_rect.width/2,y-text_rect.height/2))
    
def writeInfo(text):
    pygame.draw.rect(screen, WHITE, info_rect, width=0)
    writeText(screen, text, board_center[0], int(board_center[1] + (N/2)*SQSIDE*1.7))
    pygame.display.update()
    
def drawSquare(screen, coords, size, angle, label, color):
    x = coords[0]
    y = coords[1]
    x1 = size/2*(math.cos(angle) - math.sin(angle))
    y1 = size/2*(math.cos(angle) + math.sin(angle))
    x2 = -y1
    y2 = x1
    x3 = -x1
    y3 = -y1
    x4 = -x2
    y4 = -y2
    pygame.draw.polygon(screen, color, [(x+x1,y+y1),(x+x2,y+y2),(x+x3,y+y3),(x+x4,y+y4)])
    pygame.draw.line(screen, BLACK, (x+x1,y+y1), (x+x2,y+y2), width=LINEWIDTH);
    pygame.draw.line(screen, BLACK, (x+x2,y+y2), (x+x3,y+y3), width=LINEWIDTH);
    pygame.draw.line(screen, BLACK, (x+x3,y+y3), (x+x4,y+y4), width=LINEWIDTH);
    pygame.draw.line(screen, BLACK, (x+x4,y+y4), (x+x1,y+y1), width=LINEWIDTH);
    writeText(screen, label, x, y)

def rotate(point, angle, center):
    x1 = point[0] - center[0]
    y1 = point[1] - center[1]
    
    x2 = x1*math.cos(angle) - y1*math.sin(angle)
    y2 = y1*math.cos(angle) + x1*math.sin(angle)
    
    newX = x2 + center[0]
    newY = y2 + center[1]
    
    return (newX, newY)
    
def rotX(board, turns = 1):
    b = len(board)
    newBoard = copy.deepcopy(board)
    for i in range(b):
        for j in range(b):
            if turns == 1:
                newBoard[i][j] = board[j][b-1-i]
                
            elif turns == 2:
                newBoard[i][j] = board[b-1-i][b-1-j]
                
            elif turns == 3:
                newBoard[i][j] = board[b-1-j][i]
            
    return newBoard

def rotY(board, turns = 1):
    b = len(board)
    newBoard = copy.deepcopy(board)
    for i in range(b):
        for j in range(1,b+1):
            if turns == 1:
                newBoard[i][j] = board[j-1][b-i]
                
            elif turns == 2:
                newBoard[i][j] = board[b-1-i][b+1-j]
                
            elif turns == 3:
                newBoard[i][j] = board[b-j][i+1]
            
    return newBoard

def defaultBoard():
    board = []
    for j in range(N):
        row = []
        for i in range(M):
            row.append(j*M + i + 1)
        
        board.append(row)

    return board
    
def defaultColors():
    boardColors = []
    for j in range(N):
        row = []
        for i in range(M):
            row.append(getCellColor(i, j, M, N))
		
        boardColors.append(row)
    
    return boardColors

board = defaultBoard()
boardColors = defaultColors()
prev_board = None
prev_boardColors = None

NO_ROT = 0
X_CCW = 1
X_CW = 2
Y_CCW = 3
Y_CW = 4
TYPE_X = 0
TYPE_Y = 1

# FOR THE AUTOSOLVE

def makeBoard(b):   # Make b+1 x b NRP board
    board = []
    num = 1
    for i in range(b):
        row = []
        for j in range(b+1):
            row.append(num)
            num += 1
        board.append(row)
        
    return board

def collapseBoard(board):  # List out elements into a 1D array, and also 0-index
    out = []
    for row in board:
        for i in row:
            out.append(i-1)
            
    return out
    
def cycleLens(arr):
    remaining = set(arr)
    output = []
    while len(remaining) > 0:
        length = 1
        num = remaining.pop()
        while 1:
            num = arr[num]
            if num not in remaining:
                break
            
            remaining.remove(num)
            length += 1
            
        output.append(length)
        
    return output
    
def isEven(board):
    cycles = cycleLens(collapseBoard(board))
    num_even_cycles = 0
    for cycle in cycles:
        if cycle%2 == 0:
            num_even_cycles += 1
            
    return num_even_cycles % 2 == 0
        

class Alg():
    def __init__(self, board, moves):
        self.board = board
        self.moves = moves
        self.M = len(board[0])
        self.N = len(board)
        
    def moveX_CCW(self):
        self.board = rotX(self.board)
        self.moves.append(X_CCW)
        return X_CCW
        
    def moveX_CW(self):
        self.board = rotX(self.board, turns=3)
        self.moves.append(X_CW)
        return X_CW
        
    def moveY_CCW(self):
        self.board = rotY(self.board)
        self.moves.append(Y_CCW)
        return Y_CCW
        
    def moveY_CW(self):
        self.board = rotY(self.board, turns=3)
        self.moves.append(Y_CW)
        return Y_CW
        
    def execute(self, moveList):
        # the fuck am i doing
        [{X_CCW : self.moveX_CCW, Y_CCW : self.moveY_CCW, X_CW : self.moveX_CW, Y_CW : self.moveY_CW}[move]() for move in moveList]
        return moveList

    def parseMoves(self, moveString):
        # Example format:  XYYX'YYXYXY'
        
        stringLen = len(moveString)
        if stringLen == 0:
            return []
            
        if moveString[stringLen-1] == "'":
            parsed = self.parseMoves(moveString[:stringLen-2])
            if moveString[stringLen-2] == "X":
                parsed.append(X_CW)
            elif moveString[stringLen-2] == "Y":
                parsed.append(Y_CW)
                
            return parsed
            
        parsed = self.parseMoves(moveString[:stringLen-1])
        if moveString[stringLen-1] == "X":
            parsed.append(X_CCW)
        elif moveString[stringLen-1] == "Y":
            parsed.append(Y_CCW)
                
        return parsed
    
    def execute_str(self, moveString):
        return self.execute(self.parseMoves(moveString))
    
    def standard_3_cycle_alg(self):
        # Sends (1,0) --> (N-1,3) --> (1,M-2) --> (1,0)
        return self.parseMoves("YXYXXYYXY'X'YXYXXYX'YYXXY'X'Y'XYYXY'XXY'X'Y'XYX'YYX'")
    
    def in_critical_region(self, coords):
        if self.N % 2 == 0:
            return all([N/2 <= coords[0], coords[0] <= N-1,
                          1 <= coords[1], coords[1] <= N/2])
               
        elif self.N % 2 == 1:
            return all([(N-1)/2 <= coords[0], coords[0] <= N-1,
                              1 <= coords[1], coords[1] <= (N+1)/2])
                              
    def in_critical_region_alt(self, coords):
        return self.in_critical_region((N-1-coords[0], coords[1]))
    
    def at_center(self, coords):
        
        if self.N % 2 == 0:
            return coords == (N/2, N/2)
            
        elif self.N % 2 == 1:
            # Case on the parity of the square
            # One-away good enough
            if self.coords_even(coords):
                return coords in  [((N-1)/2+x, (N+1)/2+y) for (x,y) in [(-1,0),(1,0),(0,-1),(0,1)]]
            else:
                return coords in  [((N-1)/2+x, (N-1)/2+y) for (x,y) in [(-1,0),(1,0),(0,-1),(0,1)]]
    
    def spiral(self, coords):
        moves = []
        num = self.num_at(coords)
        
        if coords[1] == 0: # First column
            moves.append(self.moveX_CCW())
            moves.append(self.moveX_CCW())
        
        while not self.at_center(self.coords(num)):
            while not self.in_critical_region(self.coords(num)):
                moves.append(self.moveY_CCW())
            if self.at_center(self.coords(num)):
                break
            moves.append(self.moveX_CCW())
            
        if self.N % 2 == 1:
            # May need to adjust
            if self.coords_even(coords):
                while self.coords(num) != ((N-1)/2,(N-1)/2):
                    moves.append(self.moveY_CCW())
                    
            else:
                while self.coords(num) != ((N-1)/2,(N+1)/2):
                    moves.append(self.moveX_CCW())
            
        return moves
        
    def spiral_alt(self, coords):
        # REQUIRES:  working on the even parity if N is odd
        
        NEUTRAL = 0
        ALT = 1
        
        moves = []
        num = self.num_at(coords)
        
        curr_mode = NEUTRAL
        
        if coords[1] == 0: # First column
            moves += self.execute_str("XYX'")
        
        while not self.at_center(self.coords(num)):
            # print('---')
            # print(self.coords(num))
            # print('mode:', curr_mode)
            if curr_mode == NEUTRAL:
                while not self.in_critical_region(self.coords(num)):
                    moves.append(self.moveY_CCW())
                    # print('Y')
                # print('in region')
                if self.at_center(self.coords(num)):
                    break
                    
                moves.append(self.moveX_CCW())
                # print('X')
                curr_mode = ALT
                
            elif curr_mode == ALT:
                while not self.in_critical_region_alt(self.coords(num)):
                    moves.append(self.moveY_CCW())
                    # print('Y')
                # print('in region')
                if self.at_center(self.coords(num)):
                    break
                    
                moves.append(self.moveX_CW())   
                # print("X'")
                curr_mode = NEUTRAL
        
        if self.N % 2 == 1:
            # May need to adjust
            # We assume self.coords_even(coords) holds

            while self.coords(num) != ((N-1)/2,(N-1)/2):
                moves.append(self.moveY_CCW())
        
        if self.N % 2 == 1:
            if curr_mode == ALT:
                moves.append(self.moveX_CW())
                
        elif self.N % 2 == 0:
            if curr_mode == ALT:
                moves += self.execute_str("Y'X'")
        
        return moves
    
    def spiral_alt_cycle(self, coords):
        # REQUIRES:  working on the even parity if N is odd
        
        NEUTRAL = 0
        CYCLED = 1
        
        mode = NEUTRAL
        
        moves = []
        num = self.num_at(coords)
        
        if coords[1] == 0: # First column
            if N%2 == 1:
                moves += self.execute_str("XYYXYXY'XXY'XY'X'") # apparently XYYX'YYXYX'YYXYYX' works too, and is more logical, just annoying
            else:
                moves += self.execute_str("XYYX'Y'XY'X'")
        
        while not self.at_center(self.coords(num)):
            # print('---')
            # print(self.coords(num))
            # print('mode:', mode)
            while not self.in_critical_region_alt(self.coords(num)):
                if mode == CYCLED:
                    moves += self.cycle()
                    if self.N == 5:
                        moves += self.cycle()
                    mode = NEUTRAL
                moves.append(self.moveY_CCW())
            if self.at_center(self.coords(num)):
                break
                
            if mode == NEUTRAL:
                moves += self.inv_cycle()
                if self.N == 5:
                    moves += self.inv_cycle()
                mode = CYCLED
            moves.append(self.moveX_CW())
            
        if self.N % 2 == 1:
            # May need to adjust
            if self.coords_even(coords):
                while self.coords(num) != ((N-1)/2,(N-1)/2):
                    if mode == CYCLED:
                        moves += self.cycle()
                        if self.N == 5:
                            moves += self.cycle()
                        mode = NEUTRAL
                    moves.append(self.moveY_CCW())
                    
            else:
                while self.coords(num) != ((N-1)/2,(N+1)/2):
                    if mode == NEUTRAL:
                        moves += self.inv_cycle()
                        if self.N == 5:
                            moves += self.inv_cycle()
                        mode = CYCLED
                    moves.append(self.moveX_CCW())
        
        if mode == CYCLED:
            moves += self.cycle()
            if self.N == 5:
                moves += self.cycle()
        
        return moves
    
    def cycle(self):
        return self.execute_str("YXYXYXYX")
        
    def inv_cycle(self):
        return self.execute_str("X'Y'X'Y'X'Y'X'Y'")
    
    def invertMove(self, move):
        return [X_CW, X_CCW, Y_CW, Y_CCW][move-1]

    def invertMoves(self, moves):
        return ([self.invertMove(move) for move in moves])[::-1]
    
    def reflect(self, coords):
        return (coords[0], self.M-1-coords[1])
        
    def reflectMove(self, move):
        return [Y_CW, Y_CCW, X_CW, X_CCW][move-1]
        
    def reflectMoves(self, moves):
        return [self.reflectMove(move) for move in moves]
    
    def num_at(self, coords):
        return self.board[coords[0]][coords[1]]
        
    def coords(self, num):
        for j in range(self.N):
            for i in range(self.M):
                if self.board[j][i] == num:
                    return (j, i)
                    
    def coords_if_solved(self, num):
        return ((num-1) // self.M, (num - 1) % self.M)
      
    def is_num_solved(self, num):
        return self.coords(num) == self.coords_if_solved(num)
        
    def is_solved(self):
        for num in range(1, self.M * self.N + 1):
            if not self.is_num_solved(num):
                return False
        
        return True
        
    def coords_even(self, coords):
        return (coords[0] + coords[1]) % 2 == 0
        
    def num_even(self, num):
        return self.coords_even(self.coords(num))
      
    def is_X(self, move):
        return move in [X_CCW, X_CW]
    
    def is_Y(self, move):
        return move in [Y_CCW, Y_CW]
        
    def is_CCW(self, move):
        return move in [X_CCW, Y_CCW]
    
    def move_type(self, move):
        if self.is_X(move):
            return TYPE_X
        else:
            return TYPE_Y
    
    def condensedFormat(self, moves):
        if len(moves) == 0:
            return []
            
        condensed = []
            
        curr_move_type = self.move_type(moves[0])
        condensed.append([curr_move_type,0])
        
        for move in moves:
            if self.move_type(move) != curr_move_type:
                curr_move_type = self.move_type(move)
                condensed.append([curr_move_type,0])
            
            if self.is_CCW(move):
                condensed[-1][1] += 1
            else:
                condensed[-1][1] -= 1
                    
        # print(condensed)
        return condensed
      
    def simplify(self):
        condensed = self.condensedFormat(self.moves)
        if len(condensed) == 0:
            return
            
        simplified = False
        while not simplified:
            simplified = True
            
            # Search for redundancy:
            
            new_condensed = []
            
            for move_tuple in condensed:
                if move_tuple[1] % 4 == 0:
                    simplified = False
                else:
                    new_condensed.append(move_tuple)
            
            condensed = copy.deepcopy(new_condensed)
            
            # Search for merging:
            
            new_condensed = []
            new_condensed.append(condensed[0])
            
            for i in range(1, len(condensed)):
                if condensed[i][0] == new_condensed[len(new_condensed)-1][0]:
                    simplified = False
                    new_condensed[len(new_condensed)-1][1] += condensed[i][1]
                else:
                    new_condensed.append(condensed[i])
                  
            condensed = copy.deepcopy(new_condensed)
                    
        # Now just "mod 4"
        
        new_moves = []
        for move_tuple in new_condensed:
            if move_tuple[1] % 4 == 1:
                if move_tuple[0] == TYPE_X:
                    new_moves.append(X_CCW)
                else:
                    new_moves.append(Y_CCW)
                    
            elif move_tuple[1] % 4 == 3:
                if move_tuple[0] == TYPE_X:
                    new_moves.append(X_CW)
                else:
                    new_moves.append(Y_CW)
                    
            elif move_tuple[1] % 4 == 2 and move_tuple[1] > 0:
                if move_tuple[0] == TYPE_X:
                    new_moves.append(X_CCW)
                    new_moves.append(X_CCW)
                else:
                    new_moves.append(Y_CCW)
                    new_moves.append(Y_CCW)
                    
            elif move_tuple[1] % 4 == 2 and move_tuple[1] < 0:
                if move_tuple[0] == TYPE_X:
                    new_moves.append(X_CW)
                    new_moves.append(X_CW)
                else:
                    new_moves.append(Y_CW)
                    new_moves.append(Y_CW)        
                    
            else:
                print('wtf')
                assert(False)
                
        self.moves = new_moves
                    
# TESTING SIMPLIFY #

# testAlg1 = Alg(defaultBoard(),[])
# testAlg1.execute_str("XXXY'Y'XYYYYYXYYYYXXXXYYYYX")
# testAlg1.simplify()
# print(testAlg1.moves)
# exit()
                    
currRotation = 0

DEFAULT_ROT_TIME = 250

ROT_TIME = DEFAULT_ROT_TIME
if AUTO_SOLVE:
    ROT_TIME = AUTO_MOVE_TIME
start_rot_time = 0

if SCRAMBLE:
    for i in range(1000):
        move = random.randint(0,3)
        if move == 0:
            board = rotX(board, turns=1)
            boardColors = rotX(boardColors, turns=1)
        if move == 1:
            board = rotX(board, turns=3)
            boardColors = rotX(boardColors, turns=3)
        if move == 2:
            board = rotY(board, turns=1)
            boardColors = rotY(boardColors, turns=1)
        if move == 3:
            board = rotY(board, turns=3)
            boardColors = rotY(boardColors, turns=3)

# 0:  full AUTOSOLVE
# 1:  See spiral in action
# 2:  Send a number to the top-left / top-right (if parity bad)

TRIAL = 0

def auto_solve(board):
    if AUTO_SOLVE:
        solving_board = copy.deepcopy(board)
        alg = Alg(solving_board, [])
        
        if TRIAL == 0:
            print('Computing solution...')
        
            # Do a random move if parity is odd
            
            if not isEven(alg.board):
                alg.moveX_CCW()
                
            # Solve each number
            
            for num in range(1, M*N+1):
                if alg.is_num_solved(num):
                    continue
                    
                num2 = alg.num_at(alg.coords_if_solved(num))
                num3 = alg.num_at(alg.coords_if_solved(num2))
                
                if num == num3:
                    # oh oops that's a 2-cycle, need to choose different num3
                    for i in range(num+1, M*N+1):
                        if not alg.is_num_solved(i) and i != num2:
                            num3 = i
                            break
                 
                if SHOW_COMPUTATION:
                    print('Cycling', num, num2, num3)
                
                badParity = (N % 2 == 1) and not alg.num_even(num)
                
                if SHOW_COMPUTATION and badParity:
                    print('Parity is bad, reflection needed')
                
                pos1 = alg.coords(num)
                pos2 = alg.coords(num2)
                pos3 = alg.coords(num3)
                
                if SHOW_COMPUTATION:
                    print(num, 'at', alg.coords(num))
                    print(num2, 'at', alg.coords(num2))
                    print(num3, 'at', alg.coords(num3))
                
                # We will execute the 3-cycle
                #        pos1 --> pos2 --> pos3 --> pos1  
                # Let's set up the intermediate and goal coords, 
                # carefully accounting for parity and reflecting if parity is bad
            
                if N % 2 == 0:
                    int1 = (0,0)
                    int2 = (1,0)
                    int2int = (N-1,M-1)
                    int3 = (N/2-1, (M-1)/2)
                    
                elif N % 2 == 1:
                    # if alg.num_even(num):
                        # int1 = (0,0)
                        # int2 = (2,0)
                        # int2int = (N-2,M-1)
                        # int3 = ((N-1)/2, M/2-1)
                    # else:
                        # int1 = alg.reflect((0,0))
                        # int2 = alg.reflect((2,0))
                        # tnt2int = alg.reflect((N-2,M))
                        # int3 = alg.reflect(((N-1)/2, M/2-1))
                    int1 = (0,0)
                    int2 = (2,0)
                    int2int = (N-2,M-1)
                    int3 = ((N-1)/2, M/2-1)
                    
                # if badParity:
                    # goal1 = (1, 0)
                    # goal2 = (N-1, 3)
                    # goal3 = (1, M-2)
                # else:
                    # goal1 = alg.reflect((1, 0))
                    # goal2 = alg.reflect((N-1, 3))
                    # goal3 = alg.reflect((1, M-2))
                    
                goal1 = alg.reflect((1, 0))
                goal2 = alg.reflect((N-1, 3))
                goal3 = alg.reflect((1, M-2))
                
                # goal1 = (0, 0)
                # goal2 = (0, 1)
                # goal3 = (0, 2)

                algDummy = Alg(solving_board, []) # When I don't want to execute the moves
                
                ### FORWARD INTERMEDIATE ###
                
                if SHOW_COMPUTATION:
                    print('Forward (===>)')
                
                forward_intermediate_moves = []
                
                # pos1 --> center via spiral
                
                if badParity:
                    moves = algDummy.spiral(alg.reflect(pos1))
                else:
                    moves = algDummy.spiral(pos1)
                
                if badParity:
                    moves = alg.reflectMoves(moves)
                    
                forward_intermediate_moves += alg.execute(moves)

                if SHOW_COMPUTATION:
                    print('executed spiral:  pos1 --> center')
                    print(num, 'at', alg.coords(num))
                    print(num2, 'at', alg.coords(num2))
                    print(num3, 'at', alg.coords(num3))
                
                # center --> int1 via spiral
                
                backMoves = algDummy.spiral(int1)
                if badParity:
                    backMoves = alg.reflectMoves(backMoves)
                    
                forward_intermediate_moves += alg.execute(alg.invertMoves(backMoves))
                
                if SHOW_COMPUTATION:
                    print('executed spiral:  center --> int 1')
                    print(num, 'at', alg.coords(num))
                    print(num2, 'at', alg.coords(num2))
                    print(num3, 'at', alg.coords(num3))
                
                # pos2 --> center via spiral alt
                
                if badParity:
                    moves = algDummy.spiral_alt(alg.reflect(alg.coords(num2)))
                else:
                    moves = algDummy.spiral_alt(alg.coords(num2))
                
                if N % 2 == 0:
                    moves.append(Y_CW)
                
                if badParity:
                    moves = alg.reflectMoves(moves)
                    
                forward_intermediate_moves += alg.execute(moves)
                
                if SHOW_COMPUTATION:
                    print('executed alternating spiral:  pos2 --> center')
                    print(num, 'at', alg.coords(num))
                    print(num2, 'at', alg.coords(num2))
                    print(num3, 'at', alg.coords(num3))
                
                # center --> int2 via spiral alt
                
                if badParity:
                    backMoves = algDummy.spiral_alt(int2int)
                else:
                    backMoves = algDummy.spiral_alt(int2int)
                # print(int2int)
                # print([["","X","X'","Y","Y'"][move] for move in backMoves])
                if N % 2 == 0:
                    backMoves.append(Y_CW)  # Now we're at int2int
                
                if badParity:
                    backMoves = alg.reflectMoves(backMoves)
                    # print([["","X","X'","Y","Y'"][move] for move in backMoves])
                forward_intermediate_moves += alg.execute(alg.invertMoves(backMoves))
                
                if badParity:
                    forward_intermediate_moves += alg.execute(alg.reflectMoves(alg.parseMoves("XY'X'")))
                else:
                    forward_intermediate_moves += alg.execute_str("XY'X'")  # This sends int2int --> int2
                
                if SHOW_COMPUTATION:
                    print('executed alternating spiral:  center --> int2')
                    print(num, 'at', alg.coords(num))
                    print(num2, 'at', alg.coords(num2))
                    print(num3, 'at', alg.coords(num3))
                
                # pos3 --> center = int3 via spiral alt cycle
                
                if badParity:
                    moves = algDummy.spiral_alt_cycle(alg.reflect(alg.coords(num3)))
                else:
                    moves = algDummy.spiral_alt_cycle(alg.coords(num3))
                
                if N % 2 == 0:
                    moves.append(Y_CW)
                
                if badParity:
                    moves = alg.reflectMoves(moves)
                    # print(moves)
                    
                forward_intermediate_moves += alg.execute(moves)
                
                if SHOW_COMPUTATION:
                    print('executed SPIRAL-CYCLE:  pos3 --> int3')
                    print(num, 'at', alg.coords(num))
                    print(num2, 'at', alg.coords(num2))
                    print(num3, 'at', alg.coords(num3))
                    # print(moves)
                
                # print([["","X","X'","Y","Y'"][move] for move in alg.moves])
                # print([["","X","X'","Y","Y'"][move] for move in forward_intermediate_moves])
                
                # print([["","X","X'","Y","Y'"][move] for move in alg.moves])
                
                ### BACKWARD INTERMEDIATE ###
                
                if SHOW_COMPUTATION:
                    print('Backward (<===)')
                
                backward_intermediate_moves = []
                backDummy = Alg(defaultBoard(),[])
                
                goalNum1 = backDummy.num_at(goal1)
                goalNum2 = backDummy.num_at(goal2)
                goalNum3 = backDummy.num_at(goal3)
                
                if SHOW_COMPUTATION:
                    print('Handling phantoms:', goalNum1, goalNum2, goalNum3)
                    print(goalNum1, 'at', backDummy.coords(goalNum1))
                    print(goalNum2, 'at', backDummy.coords(goalNum2))
                    print(goalNum3, 'at', backDummy.coords(goalNum3))
                         
                # goal1 --> center via spiral
                
                backward_intermediate_moves += backDummy.spiral(goal1)
                
                if SHOW_COMPUTATION:
                    print('Executed spiral: goal1 --> center')
                    print(goalNum1, 'at', backDummy.coords(goalNum1))
                    print(goalNum2, 'at', backDummy.coords(goalNum2))
                    print(goalNum3, 'at', backDummy.coords(goalNum3))

                # print([["","X","X'","Y","Y'"][move] for move in backDummy.moves])
                # print([["","X","X'","Y","Y'"][move] for move in backward_intermediate_moves])
                
                # center --> int1 via spiral
                
                backMoves = algDummy.spiral(int1)
                
                backward_intermediate_moves += backDummy.execute(alg.invertMoves(backMoves))
                
                if SHOW_COMPUTATION:
                    print('Executed spiral: center --> int1')
                    print(goalNum1, 'at', backDummy.coords(goalNum1))
                    print(goalNum2, 'at', backDummy.coords(goalNum2))
                    print(goalNum3, 'at', backDummy.coords(goalNum3))
                    
                # goal2 --> center via spiral alt
                
                moves = algDummy.spiral_alt(backDummy.coords(goalNum2))
                
                if N % 2 == 0:
                    moves.append(Y_CW)
                
                # if badParity:
                    # moves = alg.reflectMoves(moves)
                    
                backward_intermediate_moves += backDummy.execute(moves)
                
                if SHOW_COMPUTATION:
                    print('Executed alternating spiral: goal2 --> center')
                    print(goalNum1, 'at', backDummy.coords(goalNum1))
                    print(goalNum2, 'at', backDummy.coords(goalNum2))
                    print(goalNum3, 'at', backDummy.coords(goalNum3))
                
                # center --> int2 via spiral alt
                
                backMoves = algDummy.spiral_alt(int2int)
                
                if N % 2 == 0:
                    backMoves.append(Y_CW)  # Now we're at int2int
                
                # if badParity:
                    # backMoves = alg.reflect(backMoves)
                    
                backward_intermediate_moves += backDummy.execute(alg.invertMoves(backMoves))
                backward_intermediate_moves += backDummy.execute_str("XY'X'")  # This sends int2int --> int2

                if SHOW_COMPUTATION:
                    print('Executed alternating spiral: center --> int2')
                    print(goalNum1, 'at', backDummy.coords(goalNum1))
                    print(goalNum2, 'at', backDummy.coords(goalNum2))
                    print(goalNum3, 'at', backDummy.coords(goalNum3))
                
                # pos3 --> center = int3 via spiral alt cycle
                
                moves = algDummy.spiral_alt_cycle(backDummy.coords(goalNum3))
                
                if N % 2 == 0:
                    moves.append(Y_CW)
                
                # if badParity:
                    # moves = alg.reflectMoves(moves)
                    
                backward_intermediate_moves += backDummy.execute(moves)
                
                if SHOW_COMPUTATION:
                    print('Executed SPIRAL-CYCLE: goal3 --> int3')
                    print(goalNum1, 'at', backDummy.coords(goalNum1))
                    print(goalNum2, 'at', backDummy.coords(goalNum2))
                    print(goalNum3, 'at', backDummy.coords(goalNum3))
                
                ## SEND TO GOALS ##
                
                if badParity:
                    alg.execute(alg.reflectMoves(alg.invertMoves(backward_intermediate_moves)))
                else:
                    alg.execute(alg.invertMoves(backward_intermediate_moves))
                
                ## EXECUTE 3-CYCLE ##
                
                if badParity:
                    alg.execute(alg.invertMoves(alg.standard_3_cycle_alg()))
                else:
                    alg.execute(alg.invertMoves(alg.reflectMoves(alg.standard_3_cycle_alg())))
                
                ## UNDO THE DAMAGE ##
                
                # if badParity:
                    # alg.execute(alg.reflectMoves(backward_intermediate_moves))
                    # alg.execute(alg.reflectMoves(alg.invertMoves(forward_intermediate_moves)))
                # else:
                    # alg.execute(backward_intermediate_moves)
                    # alg.execute(alg.invertMoves(forward_intermediate_moves))
                if badParity:
                    alg.execute(alg.reflectMoves(backward_intermediate_moves))
                else:
                    alg.execute(backward_intermediate_moves)
                
                alg.execute(alg.invertMoves(forward_intermediate_moves))
                
                if SHOW_COMPUTATION:
                    print('3-cycle plan executed')
                    print('---------------------')
                
            assert alg.is_solved()
            print('SOLUTION VERIFIED')
            writeInfo('Solution Found!  Simplifying move list...')
                
        elif TRIAL == 1:
            alg.spiral(alg.coords(1))
            
        elif TRIAL == 2:
            num = 22
            if N % 2 == 1:
                if alg.num_even(num):
                    corner = (0,0)
                else:
                    corner = (0,M-1)
            else:
                corner = (0,0)
               
            alg.spiral(alg.coords(num))

            alg2 = Alg(solving_board,[])
            backSpiralMoves = alg2.spiral(corner)
            alg.execute(alg.invertMoves(backSpiralMoves))
        
            # alg.spiral(corner)
            # alg.execute(backSpiralMoves)
            
        # Simplify

        before_length = len(alg.moves)
        alg.simplify()
        after_length = len(alg.moves)
        print(before_length - after_length, 'moves cut')
            
        solution = alg.moves
            
        solution.reverse() # View as stack
        
        print('Solution Length:', len(solution))
        print('Autosolving...')
        writeInfo('Autosolving...')
        
        return solution

screen.fill(WHITE)

show_instructions = False
if not AUTO_SOLVE:
    show_instructions = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            if show_instructions:
                show_instructions = False
                writeInfo('')
                
            if not AUTO_SOLVE:                
                if event.key in [K_a, K_s, K_k, K_l]:
                    prev_board = copy.deepcopy(board)
                    prev_boardColors = copy.deepcopy(boardColors)
                    start_rot_time = pygame.time.get_ticks()
                    
                if event.key == K_a:
                    board = rotX(board)
                    boardColors = rotX(boardColors)
                    currRotation = X_CCW

                if event.key == K_s:
                    board = rotX(board, turns=3)
                    boardColors = rotX(boardColors, turns=3)
                    currRotation = X_CW
                    
                if event.key == K_k:
                    board = rotY(board)
                    boardColors = rotY(boardColors)
                    currRotation = Y_CCW
                    
                if event.key == K_l:
                    board = rotY(board, turns=3)
                    boardColors = rotY(boardColors, turns=3)
                    currRotation = Y_CW
                
                if event.key == K_r:
                    board = defaultBoard()
                    boardColors = defaultColors()
                    
                if event.key == K_x:
                    AUTO_SOLVE = True
                    ROT_TIME = AUTO_MOVE_TIME
                    writeInfo('Autosolving...')
                    solution = auto_solve(board)
                    last_move_time = pygame.time.get_ticks()
                
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()

    if AUTO_SOLVE:
        if pygame.time.get_ticks() - last_move_time >= AUTO_MOVE_TIME:
            if len(solution) == 0:
                AUTO_SOLVE = False
                ROT_TIME = DEFAULT_ROT_TIME
                writeInfo('Solved!')
                continue
                
            last_move_time = pygame.time.get_ticks()
            move = solution.pop()
            
            # spaghetti code yay
            currRotation = move
            prev_board = copy.deepcopy(board)
            prev_boardColors = copy.deepcopy(boardColors)
            start_rot_time = pygame.time.get_ticks()
                
            if move == X_CCW:
                board = rotX(board)
                boardColors = rotX(boardColors)

            if move == X_CW:
                board = rotX(board, turns=3)
                boardColors = rotX(boardColors, turns=3)
                
            if move == Y_CCW:
                board = rotY(board)
                boardColors = rotY(boardColors)
                
            if move == Y_CW:
                board = rotY(board, turns=3)
                boardColors = rotY(boardColors, turns=3)

    pygame.draw.rect(screen, WHITE, NRP_rect,width=0)

    # Draw board
	
    top = board_center[1] - N*SQSIDE/2
    left = board_center[0] - M*SQSIDE/2
    X_CENTER = (board_center[0] - SQSIDE/2, board_center[1])
    Y_CENTER = (board_center[0] + SQSIDE/2, board_center[1])
    
    if currRotation == NO_ROT:
        for j in range(N):
            for i in range(M):
                coords = (left+SQSIDE*i+SQSIDE/2, top + SQSIDE*j+SQSIDE/2)
                drawSquare(screen, coords, SQSIDE, 0, str(board[j][i]), boardColors[j][i])
                
                # pygame.draw.rect(screen, boardColors[j][i], Rect(left+SQSIDE*i, top + SQSIDE*j, SQSIDE, SQSIDE), width=0)
                # pygame.draw.rect(screen, BLACK, Rect(left+SQSIDE*i, top + SQSIDE*j, SQSIDE, SQSIDE), width=2)
                # writeText(screen, str(board[j][i]), left+SQSIDE*i+SQSIDE/2, top + SQSIDE*j+SQSIDE/2)

    else:
        angle = (pygame.time.get_ticks() - start_rot_time) / ROT_TIME * (math.pi/2)
        
        if currRotation == X_CCW or currRotation == X_CW:
            center = X_CENTER
        elif currRotation == Y_CCW or currRotation == Y_CW:
            center = Y_CENTER 
        
        # Draw the stationary squares first
        
        if currRotation == X_CCW or currRotation == X_CW:
            i = M-1
        
        if currRotation == Y_CCW or currRotation == Y_CW:
            i = 0
            
        for j in range(N):
            coords = (left+SQSIDE*i+SQSIDE/2, top + SQSIDE*j+SQSIDE/2)
            drawSquare(screen, coords, SQSIDE, 0, str(prev_board[j][i]), prev_boardColors[j][i])
        
        # Now draw the moving squares, they'll appear on top
        
        for j in range(N):
            for i in range(M):
                coords = (left+SQSIDE*i+SQSIDE/2, top + SQSIDE*j+SQSIDE/2)
                if currRotation == X_CCW:
                    if i < M-1:
                        drawSquare(screen, rotate(coords, -angle, center), SQSIDE, -angle, str(prev_board[j][i]), prev_boardColors[j][i])
                        
                if currRotation == X_CW:
                    if i < M-1:
                        drawSquare(screen, rotate(coords, angle, center), SQSIDE, angle, str(prev_board[j][i]), prev_boardColors[j][i])
                        
                if currRotation == Y_CCW:
                    if i > 0:
                        drawSquare(screen, rotate(coords, -angle, center), SQSIDE, -angle, str(prev_board[j][i]), prev_boardColors[j][i])
                        
                if currRotation == Y_CW:
                    if i > 0:
                        drawSquare(screen, rotate(coords, angle, center), SQSIDE, angle, str(prev_board[j][i]), prev_boardColors[j][i])
        
        if pygame.time.get_ticks() - start_rot_time + 60 > ROT_TIME:
            currRotation = NO_ROT
    
    if show_instructions:
        writeInfo('Press A, S, K, L to move, R to restart, X to autosolve')
    
    clock = pygame.time.Clock()
    clock.tick(60)

    pygame.display.update()
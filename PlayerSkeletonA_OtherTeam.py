'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.

'''

import BC_state_etc as BC
import BC_checker as validator
import time

directions = {
    'up': (1, 0),
    'down': (-1, 0),
    'left': (0, -1),
    'right': (0, 1),
    'up_left': (1, -1),
    'up_right': (1, 1),
    'down_left': (-1, -1),
    'down_right': (-1, 1)
}

def nickname():
    return "Newman"


def introduce():
    return "I'm Newman Barry, a newbie Baroque Chess agent."


def prepare(player2Nickname):
    ''' Here the game master will give your agent the nickname of
    the opponent agent, in case your agent can use it in some of
    the dialog responses.  Other than that, this function can be
    used for initializing data structures, if needed.'''
    pass


def basicStaticEval(state):
    '''Use the simple method for state evaluation described in the spec.
    This is typically used in parameterized_minimax calls to verify
    that minimax and alpha-beta pruning work correctly.'''

    return staticEval(state)


def staticEval(state):
    '''Compute a more thorough static evaluation of the given state.
    This is intended for normal competitive play.  How you design this
    function could have a significant impact on your player's ability
    to win games.'''
    weights = {
        'k' : 100,
        'K' : 100,
        'p' : 1,
        'P' : 1,
        'l' : 10,
        'L' : 10,
        'i' : 1,
        'I' : 1,
        'W' : 5,
        'w' : 5,
        'f' : 4,
        'F' : 4,
        'c' : 3,
        'C' : 3,
    }

    # Generate a "static eval" based on the number of pieces on the board, weights depending on piece type,
    # and the location of the pieces on the board
    count = 0
    board = state.board
    for y in range(len(board)):
        for x in range(len(board[y])):
            piece = board[y][x]
            if piece != 0:
                piece_weight = weights[BC.CODE_TO_INIT[piece]]
                if BC.who(piece) == BC.WHITE:
                    count += piece_weight
                else:
                    count -= piece_weight

                # Encourage the squares to move near the center by adding a value representing the distance from
                # the center of the board
                if BC.who(piece) == BC.WHITE:
                    count -= y / 20
                else:
                    count += y / 20

    return count

def parameterized_minimax(currentState, alphaBeta=False, ply=3, \
                          useBasicStaticEval=True, useZobristHashing=False, \
                            alpha = float(-100000), beta = float(100000)):
    result = {
        'CURRENT_STATE_VAL': staticEval(currentState),
        'N_STATES_EXPANDED': 0,
        'N_STATIC_EVALS': 1,
        'N_CUTOFFS': 0
    }
    if ply == 0:
        return result
    if currentState.whose_move == BC.WHITE:
        provisional = -100000
    else:
        provisional = 100000

    for y in range(len(currentState.board)):
        y = 7 - y
        for x in range(len(currentState.board[y])):
            for m in get_moves(currentState, (y, x)):
                result['N_STATES_EXPANDED'] += 1
                successor = move(currentState, m)
                if successor is not None:
                    new_result = parameterized_minimax(successor, alphaBeta, ply - 1, \
                                    useBasicStaticEval, useZobristHashing, \
                                    alpha, beta)
                    new_val = new_result['CURRENT_STATE_VAL']
                    result['N_STATIC_EVALS'] += new_result['N_STATIC_EVALS']
                    result['N_STATES_EXPANDED'] += new_result['N_STATES_EXPANDED']
                    if (currentState.whose_move == BC.WHITE and new_val > provisional) \
                            or (currentState.whose_move == BC.BLACK and new_val < provisional):
                        provisional = new_val
                    if alphaBeta:
                        if currentState.whose_move == BC.WHITE:
                            alpha = max(alpha, provisional)
                        else:
                            beta = min(beta, provisional)
                        if beta <= alpha:
                            result['N_CUTOFFS'] += 1
                            break

    result['CURRENT_STATE_VAL'] = provisional
    return result


def makeMove(currentState, mcurrentRemark, timelimit=10):
    # Compute the new state for a move.
    # You should implement an anytime algorithm based on IDDFS.

    # Make up a new remark
    newRemark = "OK"
    start = time.time()

    # generate dict of possible moves
    moves = {}
    for y in range(len(currentState.board)):
        y = 7 - y
        for x in range(len(currentState.board[y])):
            for m in get_moves(currentState, (y, x)):
                successor = move(currentState, m)
                if successor is not None:
                    moves[m] = staticEval(successor)

    # IDDFS
    depth = 1
    while time.time() - start < timelimit - 0.1:
        for m in moves:
            moves[m] = DLS(move(currentState, m), depth, timelimit, start)
            if time.time() - start > timelimit - 0.1:
                break
        depth += 1


    # find best move from calculated values
    best = None
    for m in moves:
        if best is None:
            best = m
        elif (currentState.whose_move == BC.WHITE and moves[m] > moves[best]) \
                or (currentState.whose_move == BC.BLACK and moves[m] < moves[best]):
            best = m
    return [[best, move(currentState, best)], newRemark]


def DLS(currentState, depth, timelimit, start, alpha = float(-100000), beta = float(100000)):
    # returns value of currentState calculated recursively using minimax
    if depth == 0:
        return staticEval(currentState)

    if currentState.whose_move == BC.WHITE:
        provisional = alpha
        provisional = max(provisional, alpha)
    else:
        provisional = beta
        provisional = min(provisional, beta)

    for y in range(len(currentState.board)):
        y = 7 - y
        for x in range(len(currentState.board[y])):
            for m in get_moves(currentState, (y, x)):
                if time.time() - start > timelimit - 0.1:
                    return provisional
                successor = move(currentState, m)
                if successor is not None:
                    new_val = DLS(successor, depth - 1, timelimit, start, alpha, beta)
                    if (currentState.whose_move == BC.WHITE and new_val > provisional) \
                            or (currentState.whose_move == BC.BLACK and new_val < provisional):
                        provisional = new_val
                    if currentState.whose_move == BC.WHITE:
                        alpha = max(alpha, provisional)
                    else:
                        beta = min(beta, provisional)
                    if beta <= alpha:
                        break

    return provisional


def move(currentState, m):
    # returns the new state after making the given move from the current state
    y, x = m[0][0], m[0][1]
    new_y, new_x = m[1][0], m[1][1]
    piece = currentState.board[y][x]
    if BC.who(piece) == currentState.whose_move: # check if piece belongs to current player
        new_state = BC.BC_state(currentState.board) # create new state
        new_state.whose_move = currentState.whose_move
        new_state.board[y][x] = 0 # remove piece from old position
        new_state.board[new_y][new_x] = piece # place piece in new position
        if piece == BC.WHITE_PINCER or piece == BC.BLACK_PINCER:
            capturePincer(new_state, ((y, x), (new_y, new_x))) # capture pincer
        if piece == BC.WHITE_LEAPER or piece == BC.BLACK_LEAPER:
            captureLeaper(new_state, ((y, x), (new_y, new_x))) # capture leaper
        if piece == BC.WHITE_WITHDRAWER or piece == BC.BLACK_WITHDRAWER:
            captureWithdrawer(new_state, ((y, x), (new_y, new_x))) # capture withdrawer
        if piece == BC.WHITE_COORDINATOR or piece == BC.BLACK_COORDINATOR:
            captureCoordinator(new_state, ((y, x), (new_y, new_x))) # capture coordinator

        new_state.whose_move = 1 - currentState.whose_move # change turn
        return new_state # return new state

    return None


def capturePincer(currentState, move):
    #print("capture pincer")
    #if whosemove is WHITE anti move is BLACk vice versa
    board = currentState.board
    y, x = move[1][0], move[1][1]

    movement = queen_paths((y, x))
    arr = ['up', 'down', 'left', 'right']
    for paths in arr:
        path = movement[paths]
        if len(path)>= 2:
            p1 = board[path[0][0]][path[0][1]]
            p2 = board[path[1][0]][path[1][1]]
            if p1 != 0 and p2 != 0 and BC.who(p1) != currentState.whose_move and BC.who(p2) == currentState.whose_move:
                board[path[0][0]][path[0][1]] = 0


def captureLeaper(currentState, move):
    #print("capture leaper")
    #if whosemove is WHITE anti move is BLACk vice versa
    board = currentState.board
    old_y, old_x = move[0][0], move[0][1]
    new_y, new_x = move[1][0], move[1][1]

    delta_y, delta_x = new_y, new_x
    if abs(new_y - old_y) > 1:
        delta_y = new_y - int((new_y - old_y) / abs(new_y - old_y))
    if abs(new_x - old_x) > 1:
        delta_x = new_x - int((new_x - old_x) / abs(new_x - old_x))

    # check if the move from old x y to xy leaps over a piece of the opposite color
    prev_piece = board[delta_y][delta_x]
    if prev_piece != 0 and BC.who(prev_piece) != currentState.whose_move:
        board[delta_y][delta_x] = 0

def captureWithdrawer(currentState, move):
    #print("capture withdrawer")
    #The Withdrawer can only capture an enemy piece that it is adjacent to, and it must have space to pull away from
    # the enemy piece, moving along the same line that it already make with the enemy piece. It must withdraw from the
    # enemy piece at least one square in order to capture it. It may withdraw a greater distance,
    # if it is not blocked by another piece.
    board = currentState.board
    old_y, old_x = move[0][0], move[0][1]
    new_y, new_x = move[1][0], move[1][1]

    delta_y, delta_x = new_y, new_x
    if abs(new_y - old_y) > 1:
        delta_y = old_y - int((new_y - old_y) / abs(new_y - old_y))
    if abs(new_x - old_x) > 1:
        delta_x = old_x - int((new_x - old_x) / abs(new_x - old_x))

    if 0 <= delta_y <= 7 and 0 <= delta_x <= 7:
        prev_piece = board[delta_y][delta_x]
        if prev_piece != 0 and BC.who(prev_piece) != currentState.whose_move:
            board[delta_y][delta_x] = 0

def captureCoordinator(currentState, move):
    #print("capture coordinator")
    board = currentState.board
    y, x = move[1][0], move[1][1]
    for kingy in range(len(board)):
        for kingx in range(len(board[kingy])):
            piece = board[kingy][kingx]
            if (piece == BC.WHITE_KING or piece == BC.BLACK_KING) and \
                    BC.who(piece) == currentState.whose_move:
                if BC.who(board[kingy][x]) != currentState.whose_move:
                    board[kingy][x] = 0
                if BC.who(board[y][kingx]) != currentState.whose_move:
                    board[y][kingx] = 0
                return


def get_moves(currentState, coords):
    board = currentState.board
    moves_list = []
    if check_frozen(board, coords):
        return []

    piece = board[coords[0]][coords[1]]
    if piece == 0:
        return []
    elif piece == BC.BLACK_PINCER or piece == BC.WHITE_PINCER:
        moves_list = pincer(board, coords)
    elif piece == BC.BLACK_COORDINATOR or piece == BC.WHITE_COORDINATOR:
        moves_list = coordinator(board, coords)
    elif piece == BC.BLACK_LEAPER or piece == BC.WHITE_LEAPER:
        moves_list = leaper(board, coords)
    elif piece == BC.BLACK_IMITATOR or piece == BC.WHITE_IMITATOR:
        moves_list = imitator(board, coords)
    elif piece == BC.BLACK_WITHDRAWER or piece == BC.WHITE_WITHDRAWER:
        moves_list = withdrawer(board, coords)
    elif piece == BC.BLACK_KING or piece == BC.WHITE_KING:
        moves_list = king(board, coords)
    else:
        moves_list = freezer(board, coords)
    moves_list.sort(key=lambda m: m[1])
    return moves_list


def king(board, coords):
    moves = []
    who = BC.who(board[coords[0]][coords[1]])
    candidates = around(coords)
    for (y, x) in candidates:
        if board[y][x] == 0 or BC.who(board[y][x]) != who:
            moves.append((coords, (y, x)))
    return moves


def pincer(board, coords):
    moves = []
    candidates = queen_paths(coords)
    rook_directions = ["up", "down", "left", "right"]

    for d in rook_directions:
        path = candidates[d]
        for (y, x) in path:
            if board[y][x] != 0:
                break
            moves.append((coords, (y, x)))
    return moves


def leaper(board, coords):
    moves = []
    who = BC.who(board[coords[0]][coords[1]])
    candidates = queen_paths(coords)

    for path in candidates.values():
        for i in range(len(path)):
            y, x = path[i][0], path[i][1]
            if board[y][x] != 0:
                # check if leap possible
                if BC.who(board[y][x]) != who and i + 2 < len(path):
                    y, x = path[i + 1][0], path[i + 1][1]
                    if board[y][x] == 0:
                        moves.append((coords, (y, x)))
                break
            else:
                moves.append((coords, (y, x)))
    return moves


def coordinator(board, coords):
    return queen(board, coords)


def imitator(board, coords):
    return queen(board, coords)


def withdrawer(board, coords):
    return queen(board, coords)


def freezer(board, coords):
    return queen(board, coords)


def check_frozen(board, coords):
    candidates = around(coords)
    who = BC.who(board[coords[0]][coords[1]])
    for (y, x) in candidates:
        if (who == BC.BLACK and board[y][x] == BC.WHITE_FREEZER) or \
                (who == BC.WHITE and board[y][x] == BC.BLACK_FREEZER):
            return True
    return False


def around(coords):
    # returns a list of coordinates around a given point
    lst = []
    for direction, new_coords in directions.items():
        y, x = coords[0], coords[1]
        y += new_coords[0]
        x += new_coords[1]
        if 0 <= x < 8 and 0 <= y < 8:
            lst.append((y, x))
    return lst


def queen(board, coords):
    moves = []
    candidates = queen_paths(coords)
    for path in candidates.values():
        i = 0
        if i < len(path):
            y, x = path[i][0], path[i][1]
            while i < len(path) and board[y][x] == 0:
                moves.append((coords, (y, x)))
                i += 1
    return moves # list of moves


def queen_paths(coords):
    # returns a list for all possible moves for a queen piece in chess
    # initialize the dictionary of moves
    moves = {"up": [], "down": [], "left": [], "right": [],
             "up_left": [], "up_right": [], "down_left": [], "down_right": []}

    for direction, new_coords in directions.items():
        y, x = coords[0], coords[1]
        while True:
            y += new_coords[0]
            x += new_coords[1]
            if x < 0 or x >= 8 or y < 0 or y >= 8:
                break
            moves[direction].append((y, x))
    return moves

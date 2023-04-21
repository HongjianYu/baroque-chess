'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.
'''

import BC_state_etc as BC

CODE_TO_VAL = {0: 0,
               2: -1, 4: -2, 6: -2, 8: -2, 10: -2, 12: -100, 14: -2,
               3: 1, 5: 2, 7: 2, 9: 2, 11: 2, 13: 100, 15: 2}

DIRECTIONS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, -1), 4: (0, 1), 5: (1, -1), 6: (1, 0), 7: (1, 1)}


def pincer(currentState, rank, file):
    if is_immobilized(currentState, rank, file):
        return []
    pass


def coordinator(currentState, rank, file):
    if is_immobilized(currentState, rank, file):
        return []
    pass


def leaper(currentState, rank, file):
    if is_immobilized(currentState, rank, file):
        return []
    pass


def imitator(currentState, rank, file):
    if is_immobilized(currentState, rank, file):
        return []
    pass


def withdrawer(currentState, rank, file):
    if is_immobilized(currentState, rank, file):
        return []
    pass


def king(currentState, rank, file):
    if is_immobilized(currentState, rank, file):
        return []
    new_states = []
    for i in range(8):
        h_dir, v_dir = DIRECTIONS[i]
        new_rank, new_file = rank + h_dir, file + v_dir
        if is_within_board_range(new_rank, new_file):
            new_code = currentState.board[new_rank][new_file]
            if not is_ally(new_code, currentState.whose_move):
                new_state = BC.BC_state(currentState.board, currentState.whose_move)
                new_state.board[rank][file] = 0
                new_state.board[new_rank][new_file] = bcs.WHITE_KING - (1 - currentState.whose_move)
                new_state.whose_move = 1 - currentState.whose_move
                new_states.append(new_state)
    return new_states


def freezer(currentState, rank, file):
    if is_immobilized(currentState, rank, file):
        return []
    pass


VAL_TO_FUNC = {2: pincer, 4: coordinator, 6: leaper, 8: imitator, 10: withdrawer, 12: king, 14: freezer,
               3: pincer, 5: coordinator, 7: leaper, 9: imitator, 11: withdrawer, 13: king, 15: freezer}


# True if the coordinate is legal
def is_within_board_range(rank, file):
    return 0 <= rank < 8 and 0 <= file < 8


# True if there is an ally in this square
def is_ally(code, whose_move):
    return code != 0 and (code - whose_move) % 2 == 0


# True if there is an enemy in this square
def is_enemy(code, whose_move):
    return code != 0 and (code - whose_move) % 2 == 1


# True if the selected piece is immobilized
def is_immobilized(currentState, rank, file):
    for i in range(8):
        h_dir, v_dir = DIRECTIONS[i]
        new_rank, new_file = rank + h_dir, file + v_dir
        if currentState.board[new_rank][new_file] - (1 - currentState.whose_move) == BC.BLACK_FREEZER:
            return True
    return False

# radix sort
def radix(states_with_moves:list)->list:
    bin0 = [[], [], [], [], [], [], []]
    bin1 = [[], [], [], [], [], [], []]
    for i in range(len(states_with_moves)):
        # in reverse order(7, 6, 5, 4, 3, 2, 1, 0)
        # states_with_moves[i][0][0]:
        # [i]: the i-th st_w_mov
        # [0]: first item in st_w_mov[i], which is a tuple
        # [0]: first item in the st_w_mov[i][0], which is the rank(1, 2, ...)
        bin0[7 - states_with_moves[i][0][0]] += states_with_moves[i]

    # traverse list
    for i in range(8):
        for j in range(len(bin0[i])):
            bin1[bin0[i][j][0][1]] += bin0[i][j]

    # final result
    for i in range(8):
        res += bin1[i]

    return res

# Possible moves in one direction, leaper has one more possible landing square behind an enemy piece
def explore_in_one_direction(currentState, rank, file, h_dir, v_dir, is_leaper):
    moves = []
    new_rank, new_file = rank + h_dir, file + v_dir
    while is_within_board_range(new_rank, new_file) and currentState.board[new_rank][new_file] == 0:
        moves.append((new_rank, new_file))
        new_rank, new_file = new_rank + h_dir, new_file + v_dir
    if is_leaper:
        next_rank, next_file = new_rank + h_dir, new_file + v_dir
        if is_within_board_range(next_rank, next_file) and currentState.board[next_rank][next_file] == 0 and \
                is_enemy(currentState.board[new_rank][new_file], currentState.whose_move):
            moves.append((next_rank, next_file))
    return moves


def minimax(currentState, stat_dict, alphaBeta=False, ply=3,
            useBasicStaticEval=True, useZobristHashing=False):
    if ply == 0:
        return stat_dict
    provisional = -100000 if currentState.whose_move == BC.WHITE else 100000


def successors(currentState):
    pass


def parameterized_minimax(currentState, alphaBeta=False, ply=3,
                          useBasicStaticEval=True, useZobristHashing=False):
    '''Implement this testing function for your agent's basic
    capabilities here.'''
    stat_dict = {"CURRENT_STATE_VAL": basicStaticEval(currentState),
                 "currentState": 0,
                 "N_STATIC_EVALS": 1,
                 "N_CUTOFFS": 0}
    minimax(currentState, stat_dict, alphaBeta, ply, useBasicStaticEval, useZobristHashing)


def makeMove(currentState, currentRemark, timelimit=10):
    # Compute the new state for a move.
    # You should implement an anytime algorithm based on IDDFS.

    # The following is a placeholder that just copies the current state.
    newState = BC.BC_state(currentState.board)

    # Fix up whose turn it will be.
    newState.whose_move = 1 - currentState.whose_move

    # Construct a representation of the move that goes from the
    # currentState to the newState.
    # Here is a placeholder in the right format but with made-up
    # numbers:
    move = ((6, 4), (3, 4))

    # Make up a new remark
    newRemark = "I'll think harder in some future game. Here's my move"

    return [[move, newState], newRemark]


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
    res = 0
    for i in range(8):
        for j in range(8):
            res += CODE_TO_VAL[state.board[i][j]]
    return res


def staticEval(state):
    '''Compute a more thorough static evaluation of the given state.
    This is intended for normal competitive play.  How you design this
    function could have a significant impact on your player's ability
    to win games.'''
    pass

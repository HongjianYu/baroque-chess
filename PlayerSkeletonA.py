'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.
'''

import BC_state_etc as BC


def pincer(new_state, rank, file, new_rank, new_file, h_dir, v_dir):
    # (r0, f0) is the enemy, (r1, f1) is the ally
    def helper_capture(r0, f0, r1, f1):
        if is_within_board_range(r0, f0) and is_within_board_range(r1, f1) \
            and is_enemy(new_state.board[r0][f0], new_state.whose_move) \
            and is_ally(new_state.board[r1][f1], new_state.whose_move):
                new_state.board[r0][f0] = 0
    # left
    helper_capture(new_rank, new_file - 1, new_rank, new_file - 2)

    # right
    helper_capture(new_rank, new_file + 1, new_rank, new_file + 2)

    # up
    helper_capture(new_rank - 1, new_file, new_rank - 2, new_file)

    # down
    helper_capture(new_rank + 1, new_file, new_rank + 2, new_file)



def coordinator(new_state, rank, file, new_rank, new_file, h_dir, v_dir):
    # find the location of the king
    king_loc = (-1, -1)
    for i in range(8):
        for j in range(8):
            # not sure whose_move is the coordinator or the oppos
            # here we take it as ally's: (if coordinator is white, then whose_move is 1)
            if new_state.board[i][j] == BC.WHITE_KIKG - (1 - new_state.whose_move):
                king_loc = (i, j)

    if is_enemy(new_state.board[new_rank][king_loc[1]], new_state.whose_move):
        new_state[new_rank][king_loc[1]] = 0
    if is_enemy(new_state.board[king_loc[0]][new_file], new_state.whose_move):
        new_state[king_loc[0]][new_file] = 0

def leaper(new_state, rank, file, new_rank, new_file, h_dir, v_dir):
    dr, df = new_rank - rank, new_file - file
    r_behind, f_behind = new_rank - h_dir, new_file - v_dir

    '''
    # diag
    if abs(dr) == abs(df):
        if is_enemy(new_state.board[r_behind][f_behind], new_state.whose_move):
            new_state.board[r_behind][f_behind] = 0

    # vertical
    if df == 0 and dr != 0:
        if is_enemy(new_state.board[r_behind][f_behind], new_state.whose_move):
            new_state.board[r_behind][f_behind] = 0

    # horizontal
    if dr == 0 and df != 0:
        if is_enemy(new_state.board[r_behind][f_behind], new_state.whose_move):
            new_state.board[r_behind][f_behind] = 0
    '''

    # combined version (this seems to work for me)
    if is_enemy(new_state.board[r_behind][f_behind], new_state.whose_move):
            new_state.board[r_behind][f_behind] = 0


def imitator(new_state, rank, file, new_rank, new_file, h_dir, v_dir):
    pass


# Withdrawer capture update
def withdrawer(new_state, rank, file, new_rank, new_file, h_dir, v_dir):
    r_behind, f_behind = rank - h_dir, file - v_dir
    if is_within_board_range(r_behind, f_behind) \
            and is_ally(new_state.board[r_behind][f_behind], new_state.whose_move):  # oppo's ally = enemy
        new_state.board[r_behind][f_behind] = 0


# King capture update
def king(new_state, rank, file, new_rank, new_file, h_dir, v_dir):
    pass  # do nothing


# Freezer capture update
def freezer(new_state, rank, file, new_rank, new_file, h_dir, v_dir):
    pass  # do nothing


CODE_TO_VAL = {0: 0,
               2: -1, 4: -2, 6: -2, 8: -2, 10: -2, 12: -100, 14: -2,
               3: 1, 5: 2, 7: 2, 9: 2, 11: 2, 13: 100, 15: 2}

DIRECTIONS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, -1), 4: (0, 1), 5: (1, -1), 6: (1, 0), 7: (1, 1)}

CODE_TO_FUNC = {2: pincer, 4: coordinator, 6: leaper, 8: imitator, 10: withdrawer, 12: king, 14: freezer,
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
    watch_imitator = currentState.board[rank][file] + (1 - currentState.whose_move) == BC.WHITE_FREEZER
    for i in range(8):
        h_dir, v_dir = DIRECTIONS[i]
        new_rank, new_file = rank + h_dir, file + v_dir
        if is_within_board_range(rank, file):
            code = currentState.board[new_rank][new_file] - (1 - currentState.whose_move)
            if code == BC.BLACK_FREEZER or (watch_imitator and code == BC.BLACK_IMITATOR):
                return True
    return False


# Possible moves in one direction
# Leaper has one more possible landing square behind an enemy piece
# King only moves one step forward
def explore_in_one_dir(currentState, rank, file, h_dir, v_dir, is_leaper, is_king):
    moves = []
    new_rank, new_file = rank + h_dir, file + v_dir

    if is_king:
        if is_within_board_range(new_rank, new_file) and \
                not is_ally(currentState.board[new_rank][new_file], currentState.whose_move):
            moves.append(((rank, file), (new_rank, new_file)))
        return moves

    # Move one step forward per looping
    while is_within_board_range(new_rank, new_file) and currentState.board[new_rank][new_file] == 0:
        moves.append(((rank, file), (new_rank, new_file)))
        new_rank, new_file = new_rank + h_dir, new_file + v_dir

    if is_leaper:
        next_rank, next_file = new_rank + h_dir, new_file + v_dir
        if is_within_board_range(next_rank, next_file) and currentState.board[next_rank][next_file] == 0 and \
                is_enemy(currentState.board[new_rank][new_file], currentState.whose_move):
            moves.append(((rank, file), (next_rank, next_file)))

    return moves


# The core function to call to generate a report on possible moves of a given piece
def move_a_piece(currentState, rank, file):
    if is_immobilized(currentState, rank, file):
        return []

    # Find what piece this is
    code = currentState[rank][file]
    # Find its corresponding capture function
    piece_func = CODE_TO_FUNC[code]

    # Initialize the direction loop
    states_with_moves = []
    directions = [1, 3, 4, 6] if piece_func == pincer else range(8)
    for i in directions:
        h_dir, v_dir = DIRECTIONS[i]

        # Get possible moves in one direction
        moves = explore_in_one_dir(currentState, rank, file, h_dir, v_dir, piece_func == leaper, piece_func == king)

        # Iterate over each move
        for move in moves:
            # Deep copy
            new_state = BC.BC_state(currentState.board, 1 - currentState.whose_move)

            new_rank, new_file = move[1]

            # Update the board
            new_state.board[rank][file] = 0
            new_state.board[new_rank][new_file] = code
            # Piece-dependent capture
            piece_func(new_state, rank, file, new_rank, new_file, h_dir, v_dir)

            states_with_moves.append((move, new_state))

    return radix(states_with_moves)


# radix sort
# states_with_moves contains elements in the format of (((from_rank, from_file), (to_rank, to_file)), newState)
def radix(states_with_moves: list) -> list:
    bin0 = [[] for i in range(8)]
    bin1 = [[] for i in range(8)]
    for i in range(len(states_with_moves)):
        # in reverse order(7, 6, 5, 4, 3, 2, 1, 0)
        # states_with_moves[i][0][0]:
        # [i]: the i-th st_w_mov
        # [0]: first item in st_w_mov[i], which is a tuple of tuples
        # [1]: second tuple within the tuple, which is the landing square coordinate
        # [0]: first item in st_w_mov[i][0][1], which is the rank(1, 2, ...)
        bin0[7 - states_with_moves[i][0][1][0]].append(states_with_moves[i])

    # traverse list
    for i in range(8):
        for j in range(len(bin0[i])):
            bin1[bin0[i][j][0][1][1]].append(bin0[i][j])

    # final result
    res = []
    for i in range(8):
        res += bin1[i]
    return res


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

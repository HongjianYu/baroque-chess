'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.
'''

import BC_state_etc as BC
is_imitator = False


def pincer(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    # To capture a pincer, an imitator cannot move in diagonal directions, or leap over an enemy
    is_not_moving_in_diag = abs(h_dir) != abs(v_dir)
    is_not_leaping = new_state.board[new_rank - h_dir][new_file - v_dir] == 0

    def pince_capture_in_one_dir(r0, f0, r1, f1):  # to capture, (r0, f0) is the enemy, (r1, f1) is the ally
        if is_within_board(r1, f1) and \
                is_enemy(new_state.board[r0][f0], new_state.whose_move) and \
                is_ally(new_state.board[r1][f1], new_state.whose_move) and \
                (not is_imitator or is_not_moving_in_diag and is_not_leaping and
                 new_state.board[r0][f0] - (1 - new_state.whose_move) == BC.BLACK_PINCER):
            new_state.board[r0][f0] = 0

    # left
    pince_capture_in_one_dir(new_rank, new_file - 1, new_rank, new_file - 2)

    # right
    pince_capture_in_one_dir(new_rank, new_file + 1, new_rank, new_file + 2)

    # up
    pince_capture_in_one_dir(new_rank - 1, new_file, new_rank - 2, new_file)

    # down
    pince_capture_in_one_dir(new_rank + 1, new_file, new_rank + 2, new_file)


def coordinator(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    # To capture a coordinator, an imitator cannot leap over an enemy
    is_not_leaping = new_state.board[new_rank - h_dir][new_file - v_dir] == 0

    def find_king():
        for i in range(8):
            for j in range(8):
                if new_state.board[i][j] + (1 - new_state.whose_move) == BC.WHITE_KING:
                    return i, j

    king_rank, king_file = find_king()

    if not is_imitator and is_enemy(new_state.board[king_rank][new_file], new_state.whose_move) or is_not_leaping and \
            new_state.board[king_rank][new_file] - (1 - new_state.whose_move) == BC.BLACK_COORDINATOR:
        new_state[king_rank][new_file] = 0

    if not is_imitator and is_enemy(new_state.board[new_rank][king_file], new_state.whose_move) or is_not_leaping and \
            new_state.board[new_rank][king_file] - (1 - new_state.whose_move) == BC.BLACK_COORDINATOR:
        new_state[new_rank][king_file] = 0


def leaper(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    r_behind, f_behind = new_rank - h_dir, new_file - v_dir
    if not is_imitator and is_enemy(new_state.board[r_behind][f_behind], new_state.whose_move) or \
            new_state.board[r_behind][f_behind] - (1 - new_state.whose_move) == BC.BLACK_LEAPER:
        new_state.board[r_behind][f_behind] = 0


def imitator(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    pincer(new_state, rank, file, new_rank, new_file, h_dir, v_dir, True)
    coordinator(new_state, rank, file, new_rank, new_file, h_dir, v_dir, True)
    leaper(new_state, rank, file, new_rank, new_file, h_dir, v_dir, True)
    withdrawer(new_state, rank, file, new_rank, new_file, h_dir, v_dir, True)
    king(new_state, rank, file, new_rank, new_file, h_dir, v_dir, True)
    freezer(new_state, rank, file, new_rank, new_file, h_dir, v_dir, True)


def withdrawer(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    # To capture a withdrawer, an imitator cannot leap over an enemy
    is_not_leaping = new_state.board[new_rank - h_dir][new_file - v_dir] == 0

    r_behind, f_behind = rank - h_dir, file - v_dir
    if is_within_board(r_behind, f_behind) and \
            (not is_imitator and is_enemy(new_state.board[r_behind][f_behind], new_state.whose_move) or
             is_not_leaping and
             new_state.board[r_behind][f_behind] - (1 - new_state.whose_move) == BC.BLACK_WITHDRAWER):
        new_state.board[r_behind][f_behind] = 0


def king(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    pass  # do nothing


def freezer(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    pass  # do nothing


CODE_TO_VAL = {0: 0,
               2: -1, 4: -2, 6: -2, 8: -2, 10: -2, 12: -100, 14: -2,
               3: 1, 5: 2, 7: 2, 9: 2, 11: 2, 13: 100, 15: 2}

DIRECTIONS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, -1), 4: (0, 1), 5: (1, -1), 6: (1, 0), 7: (1, 1)}

CODE_TO_FUNC = {2: pincer, 4: coordinator, 6: leaper, 8: imitator, 10: withdrawer, 12: king, 14: freezer,
                3: pincer, 5: coordinator, 7: leaper, 9: imitator, 11: withdrawer, 13: king, 15: freezer}


# True if the coordinate is legal
def is_within_board(rank, file):
    return 0 <= rank < 8 and 0 <= file < 8


# True if there is an ally in this square
def is_ally(code, whose_move):
    return code != 0 and (code - whose_move) % 2 == 0


# True if there is an enemy in this square
def is_enemy(code, whose_move):
    return code != 0 and (code - whose_move) % 2 == 1


# True if the selected piece is immobilized
def is_immobilized(currentState, rank, file):
    is_freezer = currentState.board[rank][file] + (1 - currentState.whose_move) == BC.WHITE_FREEZER
    for i in range(8):
        h_dir, v_dir = DIRECTIONS[i]
        new_rank, new_file = rank + h_dir, file + v_dir
        if is_within_board(rank, file):
            code = currentState.board[new_rank][new_file] - (1 - currentState.whose_move)
            if code == BC.BLACK_FREEZER or (is_freezer and code == BC.BLACK_IMITATOR):
                return True
    return False


# Possible moves in one direction
# Leaper has one more possible landing square behind an enemy piece
# King only moves one step forward
def explore_in_one_dir(currentState, rank, file, h_dir, v_dir):
    piece_func = CODE_TO_FUNC[currentState[rank][file]]
    is_leaper, is_imitator, is_king = piece_func == leaper, piece_func == imitator, piece_func == king

    moves = []
    new_rank, new_file = rank + h_dir, file + v_dir

    # Move by a king, to capture or not
    if is_king:
        if is_within_board(new_rank, new_file) and \
                not is_ally(currentState.board[new_rank][new_file], currentState.whose_move):
            moves.append(((rank, file), (new_rank, new_file)))
        return moves

    # Move to an empty square, to capture or not
    while is_within_board(new_rank, new_file) and currentState.board[new_rank][new_file] == 0:
        moves.append(((rank, file), (new_rank, new_file)))
        new_rank, new_file = new_rank + h_dir, new_file + v_dir

    # Move to capture a king by an imitator
    if is_imitator and is_within_board(new_rank, new_file) and \
            new_rank - rank == h_dir and new_file - file == v_dir and \
            currentState.board[new_rank][new_file] - (1 - currentState.whose_move) == BC.BLACK_KING:
        moves.append(((rank, file), (new_rank, new_file)))

    # Move to capture a piece by a leaper, or to capture a leaper by an imitator
    if is_leaper or is_imitator:
        next_rank, next_file = new_rank + h_dir, new_file + v_dir
        if is_within_board(next_rank, next_file) and currentState.board[next_rank][next_file] == 0 and \
                (is_leaper and is_enemy(currentState.board[new_rank][new_file], currentState.whose_move) or
                 currentState.board[new_rank][new_file] - (1 - currentState.whose_move) == BC.BLACK_LEAPER):
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
        moves = explore_in_one_dir(currentState, rank, file, h_dir, v_dir)

        # Iterate over each move
        for move in moves:
            # Deep copy
            new_state = BC.BC_state(currentState.board, currentState.whose_move)

            new_rank, new_file = move[1]

            # Update the board
            new_state.board[rank][file] = 0
            new_state.board[new_rank][new_file] = code
            # Piece-dependent capture
            piece_func(new_state, rank, file, new_rank, new_file, h_dir, v_dir, False)
            currentState.whose_move = 1 - currentState.whose_move

            states_with_moves.append((move, new_state))

    return radix_sort(states_with_moves)


# states_with_moves contains elements in the format of (((from_rank, from_file), (to_rank, to_file)), newState)
def radix_sort(states_with_moves: list) -> list:
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
    whose_move = currentState.whose_move
    provisional = -100000 if whose_move == BC.WHITE else 100000

    # all possible moves that one side can take (move + capture)
    for s in successors(currentState):
        # update stat_dict based on the new state s
        stat_dict['CURRENT_STATE_VAL'] = basicStaticEval(s)
        stat_dict['N_STATES_EXPANDED'] += 1
        stat_dict['N_STATIC_EVALS'] += 1

        # Not sure if the updated stat_dict since it's already updated
        stat_dict = minimax(s, stat_dict, False, ply - 1, True, False)

        if whose_move == BC.WHITE and stat_dict['CURRENT_STATE_VAL'] > provisional \
        or whose_move == BC.BLACK and stat_dict['CURRENT_STATE_VAL'] < provisional:
            provisional = stat_dict['CURRENT_STATE_VAL']

        return stat_dict


def successors(currentState):
    # all possible moves that one side can take (move + capture)
    all_states_with_moves = []
    if currentState.whose_move == BC.WHITE: # WHITE's turn
        for i in range(8):
            for j in range(8):
                if currentState.board[i][j] % 2 == 1: # WHITE piece
                    all_states_with_moves.append(move_a_piece(currentState, i, j))
    else: # BLACK's turn
        for i in range(8):
            for j in range(8):
                if currentState.board[i][j] % 2 == 0: # BLACK piece
                    all_states_with_moves.append(move_a_piece(currentState, i, j))

    # all possible moves in the order of the new location of the moved piece
    return radix(all_states_with_moves)


def parameterized_minimax(currentState, alphaBeta=False, ply=3,
                          useBasicStaticEval=True, useZobristHashing=False):
    '''Implement this testing function for your agent's basic
    capabilities here.'''
    stat_dict = {"CURRENT_STATE_VAL": basicStaticEval(currentState),
                 "N_STATES_EXPANDED": 0,
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

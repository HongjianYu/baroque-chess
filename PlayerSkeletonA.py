'''PlayerSkeletonA.py
The beginnings of an agent that might someday play Baroque Chess.
Switched-On Bach by Runying Chen and Hongjian Yu, Apr 24, 2023
'''

import BC_state_etc as BC
import threading

IMITATOR_CAPTURES_IMPLEMENTED = None
NUM_OPTIONS = [0]
player2 = None


def pincer(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    # To capture a pincer, an imitator cannot move in diagonal directions
    is_not_moving_in_diag = abs(h_dir) != abs(v_dir)

    def pince_capture_in_one_dir(r0, f0, r1, f1):  # to capture, (r0, f0) is the enemy, (r1, f1) is the ally
        if is_within_board(r1, f1) and \
                is_enemy(new_state.board[r0][f0], new_state.whose_move) and \
                is_ally(new_state.board[r1][f1], new_state.whose_move) and \
                (not is_imitator or is_not_moving_in_diag and
                 new_state.board[r0][f0] - (1 - new_state.whose_move) == BC.BLACK_PINCER):
            new_state.board[r0][f0] = 0

    # left
    pince_capture_in_one_dir(new_rank, new_file - 1, new_rank, new_file - 2)

    # down
    pince_capture_in_one_dir(new_rank + 1, new_file, new_rank + 2, new_file)

    # up
    pince_capture_in_one_dir(new_rank - 1, new_file, new_rank - 2, new_file)

    # right
    pince_capture_in_one_dir(new_rank, new_file + 1, new_rank, new_file + 2)


def coordinator(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    def find_king():
        for i in range(8):
            for j in range(8):
                if new_state.board[i][j] + (1 - new_state.whose_move) == BC.WHITE_KING:
                    return i, j

    king_rank, king_file = find_king()

    if not is_imitator and is_enemy(new_state.board[king_rank][new_file], new_state.whose_move) or \
            new_state.board[king_rank][new_file] - (1 - new_state.whose_move) == BC.BLACK_COORDINATOR:
        new_state[king_rank][new_file] = 0

    if not is_imitator and is_enemy(new_state.board[new_rank][king_file], new_state.whose_move) or \
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


# Used when imitator capture is disabled
def dummy_imitator(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    pass  # do nothing


def withdrawer(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    r_behind, f_behind = rank - h_dir, file - v_dir
    if is_within_board(r_behind, f_behind) and \
            (not is_imitator and is_enemy(new_state.board[r_behind][f_behind], new_state.whose_move) or
             new_state.board[r_behind][f_behind] - (1 - new_state.whose_move) == BC.BLACK_WITHDRAWER):
        new_state.board[r_behind][f_behind] = 0


def king(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    pass  # do nothing


def freezer(new_state, rank, file, new_rank, new_file, h_dir, v_dir, is_imitator):
    pass  # do nothing


CODE_TO_FUNC = {2: pincer, 4: coordinator, 6: leaper, 8: imitator, 10: withdrawer, 12: king, 14: freezer,
                3: pincer, 5: coordinator, 7: leaper, 9: imitator, 11: withdrawer, 13: king, 15: freezer}

BASIC_CODE_TO_VAL = {0: 0,
                     2: -1, 4: -2, 6: -2, 8: -2, 10: -2, 12: -100, 14: -2,
                     3: 1, 5: 2, 7: 2, 9: 2, 11: 2, 13: 100, 15: 2}

CODE_TO_VAL = {0: 0,
               2: -3, 4: -8, 6: -6, 8: -6, 10: -10, 12: -1000, 14: -12,
               3: 3, 5: 8, 7: 6, 9: 6, 11: 10, 13: 1000, 15: 12}

DIRECTIONS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, -1), 4: (0, 1), 5: (1, -1), 6: (1, 0), 7: (1, 1)}


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
    imitator_on = CODE_TO_FUNC[BC.BLACK_IMITATOR] == imitator
    for i in range(8):
        h_dir, v_dir = DIRECTIONS[i]
        new_rank, new_file = rank + h_dir, file + v_dir
        if is_within_board(new_rank, new_file):
            code = currentState.board[new_rank][new_file] - (1 - currentState.whose_move)
            if code == BC.BLACK_FREEZER or (is_freezer and imitator_on and code == BC.BLACK_IMITATOR):
                return True
    return False


# Possible moves in one direction
# Leaper has one more possible landing square behind an enemy piece
# King only moves one step at max
def explore_in_one_dir(currentState, rank, file, h_dir, v_dir):
    piece_func = CODE_TO_FUNC[currentState.board[rank][file]]
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


# The core function to generate possible moves of a given piece
def move_a_piece(currentState, rank, file):
    if is_immobilized(currentState, rank, file):
        return []

    # Find what piece this is
    code = currentState.board[rank][file]
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


# Sort items in the format of (((from_rank, from_file), (to_rank, to_file)), newState)
def radix_sort(states_with_moves: list) -> list:
    print([item[0] for item in states_with_moves])
    # Two sets of buckets
    bin0, bin1 = [[] * 8], [[] * 8]

    # Order by ranks
    for item in states_with_moves:
        # indexed reversely: 7, 6, 5, 4, 3, 2, 1, 0
        # item[0][1][0]:
        # [0]: a tuple of two coordinates
        # [1]: the landing square coordinate
        # [0]: the landing square rank
        bin0[7 - item[0][1][0]].append(item)

    # Order by files
    for item in [item for bucket in bin0 for item in bucket]:
        # item[0][1][1]:
        # [0]: a tuple of two coordinates
        # [1]: the landing square coordinate
        # [0]: the landing square file
        bin1[item[0][1][1]].append(item)

    return [item for bucket in bin1 for item in bucket]


def minimax(currentState, stat_dict, alphaBeta=False, ply=3,
            useBasicStaticEval=True, useZobristHashing=False):
    if ply == 0:
        # Evaluate the leaf of the expansion
        stat_dict['N_STATIC_EVALS'] += 1
        return basicStaticEval(currentState) if useBasicStaticEval else staticEval(currentState)

    whose_move = currentState.whose_move
    provisional = -5000 if whose_move == BC.WHITE else 5000

    # Expand the state
    stat_dict['N_STATES_EXPANDED'] += 1
    ss = successors(currentState)
    NUM_OPTIONS[0] = len(ss)
    for s in ss[1]:

        new_val = minimax(s, stat_dict, alphaBeta, ply - 1, useBasicStaticEval, useZobristHashing)

        if whose_move == BC.WHITE and new_val > provisional \
                or whose_move == BC.BLACK and new_val < provisional:
            provisional = new_val

    return provisional


def parameterized_minimax(currentState, alphaBeta=False, ply=3,
                          useBasicStaticEval=True, useZobristHashing=False):
    '''Implement this testing function for your agent's basic
    capabilities here.'''
    stat_dict = {"CURRENT_STATE_VAL": None, "N_STATES_EXPANDED": 0, "N_STATIC_EVALS": 0, "N_CUTOFFS": 0}
    stat_dict['CURRENT_STATE_VAL'] = minimax(currentState, stat_dict, alphaBeta, ply,
                                             useBasicStaticEval, useZobristHashing)
    return stat_dict


def makeMove(currentState, currentRemark, timelimit=10):
    # Make the best decision before timeout.
    best_move = [[((-1, -1), (-1, -1)), currentState], "Something went off."]

    def move():
        whose_move = currentState.whose_move
        ss = successors(currentState)

        for i in range(8):
            appointed_move = best_move
            best_val = -5000 if whose_move == BC.WHITE else 5000

            for s in ss:
                stat_dict = parameterized_minimax(currentState, False, i, True, False)
                val = stat_dict['CURRENT_STATE_VAL']

                if whose_move == BC.WHITE and val > best_val \
                        or whose_move == BC.BLACK and val < best_val:
                    best_val = val
                    appointed_move = [s[0], s[1]]

            best_move[0] = appointed_move
            best_move[1] = "Okay, " + print_move(appointed_move[0][0]) + f". Imperfect but bizarre, {player2}."

    def print_move(movement):
        (from_rank, from_file), (to_rank, to_file) = movement
        return str(chr(ord('a') + from_file)) + str(from_rank) + str(chr(ord('a') + to_file)) + str(to_rank)

    class MoveThread:
        def __init__(self):
            self.best_shot = best_move

        def run(self, timeout):
            thread = threading.Thread(target=move)
            thread.start()
            thread.join(timeout)
            return self.best_shot

    move_thread = MoveThread()
    return move_thread.run(timelimit)


def successors(currentState):
    # Item format: (((from_rank, from_file), (to_rank, to_file)), newState)
    all_states_with_moves = []

    for j in range(8):
        for i in range(7, -1, -1):
            if is_ally(currentState.board[i][j], currentState.whose_move):
                all_states_with_moves += move_a_piece(currentState, i, j)

    return all_states_with_moves


def nickname():
    return "Switched-On Bach"


def introduce():
    return "Make Baroque Great Again."


def prepare(player2Nickname):
    '''Here the game master will give your agent the nickname of
    the opponent agent, in case your agent can use it in some of
    the dialog responses.  Other than that, this function can be
    used for initializing data structures, if needed.'''
    global player2
    player2 = player2Nickname

    # This agent implements the imitator move generator
    global IMITATOR_CAPTURES_IMPLEMENTED
    IMITATOR_CAPTURES_IMPLEMENTED = True


def enable_imitator_captures(status=False):
    if status:
        CODE_TO_FUNC[BC.WHITE_IMITATOR] = imitator
        CODE_TO_FUNC[BC.BLACK_IMITATOR] = imitator
    else:
        CODE_TO_FUNC[BC.WHITE_IMITATOR] = dummy_imitator
        CODE_TO_FUNC[BC.BLACK_IMITATOR] = dummy_imitator


def basicStaticEval(state):
    '''Use the simple method for state evaluation described in the spec.
    This is typically used in parameterized_minimax calls to verify
    that minimax and alpha-beta pruning work correctly.'''
    return sum([BASIC_CODE_TO_VAL[code] for row in state.board for code in row])


def staticEval(state):
    '''Compute a more thorough static evaluation of the given state.
    This is intended for normal competitive play.  How you design this
    function could have a significant impact on your player's ability
    to win games.'''

    # for approximation of the number of options
    expect = lambda p, n: (p - 1)*((1 - p)**n - 1) / p

    # number of pieces on the board
    n = 0
    for i in range(8):
        for j in range(8):
            if state.board[i][j] != 0: n += 1
    p = 1 / (n - 1)

    # this staticEval takes the number of options(expected) into account
    alpha = 0.5 # weight of the basicStaticEval in the new staticEval
    '''
    opts = 0.0 # number of options
    for i in range(8):
        for j in range(8):
            if state.board[i][j] % 2 == state.whose_move:
                opts += expect(p, i) + expect(p, 7 - i) # horizontal
                opts += expect(p, j) + expect(p, 7 - j) # vertical
                # diag
                opts += expect(p, min(i, j))
                opts += expect(p, min(7 - i, j))
                opts += expect(p, min(i, 7 - j))
                opts += expect(p, min(7 - i, 7 - j))

    res = alpha * basicStaticEval(state) - (1 - alpha) * opts
    '''
    res = alpha * basicStaticEval(state) - (1 - alpha) * NUM_OPTIONS

    return res

    # return sum([CODE_TO_VAL[code] for row in state.board for code in row]) * len(successors(state))

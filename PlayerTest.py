import PlayerSkeletonA as A
import BC_state_etc as BC

current = BC.parse('''
c l i w k i - f
p p - p p - p p
- - p - - - - -
- - P l - p - -
- - - I - - - -
- - - - - - - -
P P - P P P P P
F L - - K I L C
''')

current_state = BC.BC_state(current, BC.WHITE)

moves = A.move_a_piece(current_state, 4, 3)
print(moves)

best_move = [((-1, -1), (-1, -1)), current_state]
best_move_eval = -5000
for move in moves:
    print(A.basicStaticEval(move[1]))
    move_eval = A.staticEval(move[1])
    print(move_eval)
    if move_eval > best_move_eval:
        best_move_eval = move_eval
        best_move = [move, BC.BC_state(move[1].board, 1 - current_state.whose_move)]
print(best_move_eval)

next_state = best_move[1]
moves = A.move_a_piece(next_state, 1, 4)
print(moves)

best_move_eval = 5000
for move in moves:
    print(A.basicStaticEval(move[1]))
    move_eval = A.staticEval(move[1])
    print(move_eval)
    if move_eval < best_move_eval:
        best_move_eval = move_eval
        best_move = [move, BC.BC_state(move[1].board, 1 - next_state.whose_move)]
print(best_move_eval)

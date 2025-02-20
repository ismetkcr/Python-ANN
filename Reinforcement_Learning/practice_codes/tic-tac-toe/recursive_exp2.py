import numpy as np

def get_state(board):
    k = 0
    h = 0
    for i in range(2):
        for j in range(2):
            v = 0
            if board[i, j] == -1:  # X is represented by -1
                v = 1
            elif board[i, j] == 1:  # O is represented by 1
                v = 2
            h += (3 ** k) * v
            k += 1
    return h

def check_game_over(board):
    for i in range(2):
        if board[i, :].sum() == -2 or board[:, i].sum() == -2:
            return True
        if board[i, :].sum() == 2 or board[:, i].sum() == 2:
            return True
    if board.trace() == -2 or np.fliplr(board).trace() == -2:
        return True
    if board.trace() == 2 or np.fliplr(board).trace() == 2:
        return True
    if not np.any(board == 0):  # Board full
        return True
    return False

def determine_winner(board):
    for i in range(2):
        if board[i, :].sum() == -2 or board[:, i].sum() == -2:
            return -1
        if board[i, :].sum() == 2 or board[:, i].sum() == 2:
            return 1
    if board.trace() == -2 or np.fliplr(board).trace() == -2:
        return -1
    if board.trace() == 2 or np.fliplr(board).trace() == 2:
        return 1
    return None

def get_state_hash_and_winner_with_boards(board, i=0, j=0):
    results = []

    for v in (0, -1, 1):  # 0 = empty, -1 = X, 1 = O
        board[i, j] = v  # Set the cell to the current value (empty, X, O)
        if j == 1:
            if i == 1:
                # Board is fully set, check the state
                state = get_state(board)
                ended = check_game_over(board)
                winner = determine_winner(board)
                results.append((state, winner, ended, board.copy()))
            else:
                results += get_state_hash_and_winner_with_boards(board, i + 1, 0)
        else:
            results += get_state_hash_and_winner_with_boards(board, i, j + 1)

    return results

# Generate all possible states with board configurations
results = get_state_hash_and_winner_with_boards(np.zeros((2, 2)))

# Print out the states and corresponding boards
for state, winner, ended, board in results:
    print(f"State: {state}, Winner: {winner}, Game Ended: {ended}")
    print(f"Board:\n{board}\n")

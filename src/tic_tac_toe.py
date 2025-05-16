def check_winner(board):
    lines = []
    lines.extend(board)
    lines.extend([[board[i][j] for i in range(3)] for j in range(3)])
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])

    for line in lines:
        if line[0] and all(cell == line[0] for cell in line):
            return line[0]
    if all(cell for row in board for cell in row):
        return 'Draw'
    return None

def minimax(board, player, depth=0, alpha=-float('inf'), beta=float('inf')):
    opponent = 'O' if player == 'X' else 'X'
    winner = check_winner(board)

    if winner == player:
        return 10 - depth, None
    if winner == opponent:
        return depth - 10, None
    if winner == 'Draw':
        return 0, None

    best_move = None

    is_maximizing = (depth % 2 == 0)
    if is_maximizing:
        best_score = -float('inf')
        current = player
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = current
                    score, _ = minimax(board, player, depth+1, alpha, beta)
                    board[i][j] = ''
                    if score > best_score:
                        best_score, best_move = score, (i, j)
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        return best_score, best_move
    else:
        best_score = float('inf')
        current = opponent
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = current
                    score, _ = minimax(board, player, depth+1, alpha, beta)
                    board[i][j] = ''
                    if score < best_score:
                        best_score, best_move = score, (i, j)
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        return best_score, best_move

    return best_score, best_move

def find_best_move(board, player):
    _, move = minimax(board, player)
    return move

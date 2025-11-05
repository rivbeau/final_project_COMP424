# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    
  
  def alpha_beta(self, board, depth, alpha, beta, maximizing_player, color, opponent):
    valid_moves = self.get_valid_unique_moves(board, color if maximizing_player else opponent)

    if depth == 0 or not valid_moves:
        return self.evaluate_board(board, color, opponent)

    if maximizing_player:
        max_eval = float('-inf')
        for move in valid_moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, color)
            eval = self.alpha_beta(new_board, depth - 1, alpha, beta, False, color, opponent)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in valid_moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, opponent)
            eval = self.alpha_beta(new_board, depth - 1, alpha, beta, True, color, opponent)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval
      
  def evaluate_board(self, board, color, opponent):
    my_pieces = np.sum(board == color)
    opp_pieces = np.sum(board == opponent)

    piece_diff = my_pieces - opp_pieces

    corners = [(0, 0), (0, board.shape[1]-1), (board.shape[0]-1, 0), (board.shape[0]-1, board.shape[1]-1)]
    corner_control = sum(1 for r, c in corners if board[r, c] == color) - sum(1 for r, c in corners if board[r, c] == opponent)

    my_mobility = len(self.get_valid_unique_moves(board, color))
    opp_mobility = len(self.get_valid_unique_moves(board, opponent))
    mobility_diff = my_mobility - opp_mobility

    return (10 * piece_diff) + (25 * corner_control) + (5 * mobility_diff)
    
  def is_1move(self, move):
    (x, y) = move.get_src();
    (xi, yi) = move.get_dest();
    return abs(x - xi) <= 1 and abs(y - yi) <= 1
  
  def get_valid_unique_moves(self, chess_board, player):
    valid_moves = get_valid_moves(chess_board, player)
    unique_destinations = set()
    unique_moves = []
    for move in valid_moves:
        if not self.is_1move(move):
            unique_moves.append(move)
            continue
        dest = move.get_dest()
        if dest not in unique_destinations:
            unique_destinations.add(dest)
            unique_moves.append(move)
    return unique_moves
  
  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).source 
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    legal_moves = self.get_valid_unique_moves(chess_board, player)
    if not legal_moves:
        return None  # No valid moves available, pass turn
    best_move = None
    best_score = float('-inf')
    
    for move in legal_moves:
        simulated_board = deepcopy(chess_board)
        execute_move(simulated_board, move, player)
        move_score = self.alpha_beta(simulated_board, 1, float('-inf'), float('inf'), False, player, opponent)
        
        if move_score > best_score:
            best_score = move_score
            best_move = move
  
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return best_move


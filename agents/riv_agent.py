# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("riv_agent")
class RivAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(RivAgent, self).__init__()
    self.name = "RivAgent"
    self.time_limit = 2.0  # seconds per move

  def step(self, chess_board, player, opponent):
      """
      Implement the step function of your agent here.
      You can use the following variables to access the chess board:
      - chess_board: a numpy array of shape (board_size, board_size)
        where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
        and 2 represents Player 2's discs (Brown).
      - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
      - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

      You should return a tuple (r,c), where (r,c) is the position where your agent
      wants to place the next disc. Use functions in helpers to determine valid moves
      and more helpful tools.

      Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
      """
      start_time = time.time()
      self.player = player
      self.opponent = opponent
      self.start_time = start_time
      
      valid_moves = get_valid_moves(chess_board, player)
      
      if not valid_moves:
          return None
      
      if len(valid_moves) == 1:
          return valid_moves[0]
      
      # Fixed search depth
      max_depth = 6
      
      # Find best move using alpha-beta
      best_move = valid_moves[0]  
      best_score = -float('inf')
      alpha = -float('inf')
      beta = float('inf')
      
      for move in valid_moves:
        # Check time limit before processing each move
        if time.time() - start_time > self.time_limit:
            print("Time limit reached, returning best move found so far")
            break
            
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, player)
        
        # Get score for this move
        score = self.min_value(new_board, alpha, beta, 1, max_depth, start_time)
        
        if score > best_score:
            best_score = score
            best_move = move
        
        alpha = max(alpha, best_score)
      
      time_taken = time.time() - start_time
      print(f"My AI's turn took {time_taken:.3f} seconds.")
      
      return best_move

  def max_value(self, chess_board, alpha, beta, current_depth, max_depth, start_time):
      """
      Maximizing player's turn in alpha-beta search.
      """
      # Check time limit first
      if time.time() - start_time > self.time_limit:
        return 0  # Return neutral score if timeout
      
      # Check if terminal state or max depth reached
      is_end, p1_score, p2_score = check_endgame(chess_board)
      if is_end or current_depth >= max_depth:
        return self.evaluate_board(chess_board, p1_score, p2_score, self.player, self.opponent)
      
      # Get valid moves for current player
      valid_moves = get_valid_moves(chess_board, self.player)
      
      # No valid moves - pass turn to opponent
      if not valid_moves:
        return self.min_value(chess_board, alpha, beta, current_depth + 1, max_depth, start_time)
      
      v = -float('inf')
      
      for move in valid_moves:
        # Check time limit for each move
        if time.time() - start_time > self.time_limit:
            break
            
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, self.player)
        
        v = max(v, self.min_value(new_board, alpha, beta, current_depth + 1, max_depth, start_time))
        
        # Alpha-beta pruning
        if v >= beta:
            return v
        alpha = max(alpha, v)
      
      return v

  def min_value(self, chess_board, alpha, beta, current_depth, max_depth, start_time):
      """
      Minimizing player's turn in alpha-beta search.
      """
      # Check time limit first
      if time.time() - start_time > self.time_limit:
        return 0  # Return neutral score if timeout
      
      # Check if terminal state or max depth reached
      is_end, p1_score, p2_score = check_endgame(chess_board)
      if is_end or current_depth >= max_depth:
        return self.evaluate_board(chess_board, p1_score, p2_score, self.player, self.opponent)
      
      # Get valid moves for opponent
      valid_moves = get_valid_moves(chess_board, self.opponent)
      
      # No valid moves - pass turn to current player
      if not valid_moves:
        return self.max_value(chess_board, alpha, beta, current_depth + 1, max_depth, start_time)
      
      v = float('inf')
      
      for move in valid_moves:
        # Check time limit for each move
        if time.time() - start_time > self.time_limit:
            break
            
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, self.opponent)
        
        v = min(v, self.max_value(new_board, alpha, beta, current_depth + 1, max_depth, start_time))
        
        # Alpha-beta pruning
        if v <= alpha:
            return v
        beta = min(beta, v)
      
      return v

  def evaluate_board(self, board,p1_score, p2_score, color, opponent):
    my_pieces = p1_score
    opp_pieces = p2_score

    piece_diff = my_pieces - opp_pieces

    corners = [(0, 0), (0, board.shape[1]-1), (board.shape[0]-1, 0), (board.shape[0]-1, board.shape[1]-1)]
    corner_control = sum(1 for r, c in corners if board[r, c] == color) - sum(1 for r, c in corners if board[r, c] == opponent)

    my_mobility = len(self.get_valid_unique_moves(board, color))
    opp_mobility = len(self.get_valid_unique_moves(board, opponent))
    mobility_diff = my_mobility - opp_mobility
    return piece_diff + mobility_diff
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
      max_depth = 3
      
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
        score = self.min_value(new_board, alpha, beta, 2, start_time)
        
        if score > best_score:
            best_score = score
            best_move = move
        
        alpha = max(alpha, best_score)
      
      time_taken = time.time() - start_time
      print(f"My AI's turn took {time_taken:.3f} seconds.")
      
      return best_move

  def max_value(self, chess_board, alpha, beta, depth, start_time):
      """
      Maximizing player's turn in alpha-beta search.
      """
      # Check time limit first
      if time.time() - start_time > self.time_limit:
        return 0  # Return neutral score if timeout
      
      # Check if terminal state or max depth reached
      is_end, p1_score, p2_score = check_endgame(chess_board)
      if is_end or depth == 0:
        return self.evaluate_board(chess_board, p1_score, p2_score, self.player, self.opponent)
      
      # Get valid moves for current player
      valid_moves = get_valid_moves(chess_board, self.player)
      
      # No valid moves - pass turn to opponent
      if not valid_moves:
        return self.min_value(chess_board, alpha, beta, depth -1, start_time)
      
      v = -float('inf')
      
      for move in valid_moves:
        # Check time limit for each move
        if time.time() - start_time > self.time_limit:
            break
            
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, self.player)
        
        v = max(v, self.min_value(new_board, alpha, beta, depth -1, start_time))
        
        # Alpha-beta pruning
        if v >= beta:
            return v
        alpha = max(alpha, v)
      
      return v

  def min_value(self, chess_board, alpha, beta, depth, start_time):
      """
      Minimizing player's turn in alpha-beta search.
      """
      # Check time limit first
      if time.time() - start_time > self.time_limit:
        return 0  # Return neutral score if timeout
      
      # Check if terminal state or max depth reached
      is_end, p1_score, p2_score = check_endgame(chess_board)
      if is_end or depth == 0:
        result = self.evaluate_board(chess_board, p1_score, p2_score, self.player, self.opponent)
        return result
      
      # Get valid moves for opponent
      valid_moves = get_valid_moves(chess_board, self.opponent)
      
      # No valid moves - pass turn to current player
      if not valid_moves:
        return self.max_value(chess_board, alpha, beta, depth -1, start_time)
      
      v = float('inf')
      
      for move in valid_moves:
        # Check time limit for each move
        if time.time() - start_time > self.time_limit:
            break
            
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, self.opponent)
        
        v = min(v, self.max_value(new_board, alpha, beta, depth -1, start_time))
        
        # Alpha-beta pruning
        if v <= alpha:
            return v
        beta = min(beta, v)
      
      return v

  def evaluate_board(self, board,p1_score, p2_score, color, opponent):
    if self.player == 1:
        my_score = p1_score
        opp_score = p2_score
    else:
        my_score = p2_score
        opp_score = p1_score
    if my_score == 0: 
        return -float('inf')
    if opp_score == 0:
        return float('inf')
    piece_diff = my_score - opp_score
    
    edges = [(0,c) for c in range(board.shape[1])] + \
        [(board.shape[0]-1,c) for c in range(board.shape[1])] + \
        [(r,0) for r in range(board.shape[0])] + \
        [(r,board.shape[1]-1) for r in range(board.shape[0])]

    edge_control = sum(1 for r,c in edges if board[r,c] == color) - sum(1 for r,c in edges if board[r,c] == opponent)
    
    # Count opponent pieces adjacent to your pieces
    adj_block = 0
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            if board[r,c] == color:
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                            if board[nr,nc] == opponent:
                                adj_block += 1

    
    centrality_bonus = 0
    size = board.shape[0]
    center = (size) // 2
    
    for r in range(size):
        for c in range(size):
            if board[r,c] == color:
                dist = abs(r - center) + abs(c - center)
                centrality_bonus += (10 - dist)
            elif board[r,c] == opponent:
                dist = abs(r - center) + abs(c - center)
                centrality_bonus -= (10 - dist)

    
    return piece_diff + 0.25*edge_control + adj_block + 0.4*centrality_bonus
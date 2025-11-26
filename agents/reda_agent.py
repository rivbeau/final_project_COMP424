# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("reda_agent")
class RedaAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent
  """

  def __init__(self):
    super(RedaAgent, self).__init__()
    self.name = "RedaAgent"
    self.time_limit = 1.99  # max time per move
    self.memo = {} # memoization table

  def get_board_hash(self, chess_board):
    """
    Create a unique hash for the board state
    Uses numpy's tobytes() to convert the board to a hashable format (bytes).
    This is the base of the memoization implementation for this agent
    """
    return chess_board.tobytes()
  
  def order_moves(self, chess_board, moves, player):
    """
    Order moves to improve alpha-beta pruning efficiency.
    
    Priority (highest to lowest):
    1. Moves that capture more opponent discs
    2. Moves to corners (strategic positions)
    3. Moves to edges (positional advantage, but not corners)
    4. Duplications (stay close, maintain presence)
    """
    if len(moves) <= 1:
      return moves
    
    board_size = chess_board.shape[0]
    opponent = 3 - player  # Quick way to get opponent: if player=1, opp=2; if player=2, opp=1
    
    def priority(move):
      dest = move.get_dest()
      src = move.get_src()
      score = 0
      
      # 1. Count captures (most important!) (100x bonus per capture)
      captures = 0
      for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
          if dr == 0 and dc == 0:
            continue
          nr, nc = dest[0] + dr, dest[1] + dc
          if 0 <= nr < board_size and 0 <= nc < board_size:
            if chess_board[nr, nc] == opponent:
              captures += 1
      score += captures * 100
      
      # 2. Corner bonus (highest priority position) (+50 bonus)
      corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
      if dest in corners:
        score += 50
      # 3. Edge bonus (only if not a corner to avoid double-counting) (+20 bonus)
      elif dest[0] == 0 or dest[0] == board_size-1 or dest[1] == 0 or dest[1] == board_size-1:
        score += 20
      
      # 4. Duplication bonus (single-tile moves keep more discs on board) (+10 bonus)
      move_distance = abs(dest[0] - src[0]) + abs(dest[1] - src[1])
      if move_distance <= 2: 
        score += 10
      
      return score
    
    # Sort moves by score (highest first)
    return sorted(moves, key=priority, reverse=True)
  
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
    self.memo.clear()  # Reset memoization table for each move
    
    valid_moves = get_valid_moves(chess_board, player)
    
    # No valid moves, passing this turn
    if not valid_moves:
      return None
    
    # If there is only one valid move, return it, no need to go through the algorithm
    if len(valid_moves) == 1:
      return valid_moves[0]
    
    
    # Searching for the best move using Alpha-Beta Pruning with ordered moves and memoization :    
    # Order moves for better alpha-beta pruning
    ordered_moves = self.order_moves(chess_board, valid_moves, player)
    
    # Searching for the best move using Alpha-Beta Pruning with memoization :    
    best_move = ordered_moves[0] # Assume first move is the best initially
    best_score = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    max_depth = 4 # Fixed max depth for the search

    for move in ordered_moves:
      # Check time limit before processing each move
      if time.time() - self.start_time >= self.time_limit:
        print("Time limit reached, returning best move found so far")
        break
      
      # Get score for this move
      new_board = deepcopy(chess_board)
      execute_move(new_board, move, player)
      score = self.min_value(new_board, alpha, beta, 1, max_depth) # after this move, it's opponent's turn, so we call min_value
      
      # Compare and update best move if needed
      if score > best_score:
        best_score = score
        best_move = move
      
      alpha = max(alpha, best_score)
    
    time_taken = time.time() - self.start_time
    print(f"My agent's turn took {time_taken:.3f} seconds.")
    
    return best_move

  def max_value(self, chess_board, alpha, beta, current_depth, max_depth):
    """
    Maximizing player's turn in alpha-beta search.
    """
    
    # Base cases of recursion :
    # Time limit check
    if time.time() - self.start_time >= self.time_limit:
      return 0  # Return neutral score if timeout
    
    # Memoization check
    board_hash = self.get_board_hash(chess_board)
    if board_hash in self.memo:
      stored_score, stored_depth = self.memo[board_hash]
      if stored_depth >= max_depth - current_depth:
        return stored_score
    
    # Check if terminal state or max depth reached
    is_end, p1_score, p2_score = check_endgame(chess_board)
    if is_end or current_depth >= max_depth:
      score = self.evaluate_board(chess_board, p1_score, p2_score) # Getting the score
      self.memo[board_hash] = (score, max_depth - current_depth)  # Store in memoization table
      return score
    
    
    # Recursive case :
    # Get valid moves for current player, if no valid moves, move to the other player (min function)
    valid_moves = get_valid_moves(chess_board, self.player)
    
    if not valid_moves:
      return self.min_value(chess_board, alpha, beta, current_depth + 1, max_depth)
    

    # Order moves for better pruning
    ordered_moves = self.order_moves(chess_board, valid_moves, self.player)
    
    v = -float('inf')
    for move in ordered_moves:
      
      # Time limit check
      if time.time() - self.start_time >= self.time_limit:
        break
          
      new_board = deepcopy(chess_board)
      execute_move(new_board, move, self.player)
      v = max(v, self.min_value(new_board, alpha, beta, current_depth + 1, max_depth))
      
      # Alpha-beta pruning
      if v >= beta:
        return v
      alpha = max(alpha, v)
    
    self.memo[board_hash] = (v, max_depth - current_depth)  # Store in memoization table
    return v

  def min_value(self, chess_board, alpha, beta, current_depth, max_depth):
    """
    Minimizing player's turn in alpha-beta search.
    """
    
    # Base cases of recursion :
    # Time limit check
    if time.time() - self.start_time >= self.time_limit:
      return 0  # Return neutral score if timeout
    
    # Memoization check
    board_hash = self.get_board_hash(chess_board)
    if board_hash in self.memo:
      stored_score, stored_depth = self.memo[board_hash]
      if stored_depth >= max_depth - current_depth:
          return stored_score
    
    # Check if terminal state or max depth reached
    is_end, p1_score, p2_score = check_endgame(chess_board)
    if is_end or current_depth >= max_depth:
      score = self.evaluate_board(chess_board, p1_score, p2_score)
      self.memo[board_hash] = (score, max_depth - current_depth)  # Store in memoization table
      return score
    
    
    # Recursive case :
    # Get valid moves for current player, if no valid moves, move to the other player (max function)
    valid_moves = get_valid_moves(chess_board, self.opponent)
    
    if not valid_moves:
      return self.max_value(chess_board, alpha, beta, current_depth + 1, max_depth)
    
    # Order moves for better pruning
    ordered_moves = self.order_moves(chess_board, valid_moves, self.opponent)
    
    v = float('inf')
    for move in ordered_moves:
      
      # Time limit check
      if time.time() - self.start_time >= self.time_limit:
        break
          
      new_board = deepcopy(chess_board)
      execute_move(new_board, move, self.opponent)
      
      v = min(v, self.max_value(new_board, alpha, beta, current_depth + 1, max_depth))
      
      # Alpha-beta pruning
      if v <= alpha:
        return v
      beta = min(beta, v)
      
    self.memo[board_hash] = (v, max_depth - current_depth)  # Store in memoization table
    return v

  def evaluate_board(self, chess_board, p1_score, p2_score):
    """
    Evaluation function based on disc count difference and "Don't Lose" heuristic.
    This evaluation method makes it so that the agent will prioritize not losing over winning big
    which will increase its chances of winning overall.
    """
    
    # p1_score and p2_score represent the number of discs each player has on the board
    if self.player == 1:
      my_score = p1_score
      opp_score = p2_score
    else:
      my_score = p2_score
      opp_score = p1_score
      
    # # Get board dimensions
    # board_size = chess_board.shape[0]  # 7
    # total_squares = board_size * board_size  # 49
    
    # # Elimination absolute wins/losses
    # if my_score == 0:
    #   return -total_squares - 1  
    # if opp_score == 0:
    #   return total_squares + 1
    
    # Simple disc difference
    disc_diff = my_score - opp_score
    
    # # Don't Lose heuristic using percentages
    # total_discs = my_score + opp_score
    # my_percentage = my_score / total_discs
    
    # # Critical danger: < 25% of discs
    # if my_percentage < 0.25:
    #   # Penalty scales from 0 to -10 as percentage drops from 25% to 0%
    #   danger_scale = (0.25 - my_percentage) / 0.25  # 0.0 to 1.0
    #   danger_penalty = -10 * danger_scale
    #   return disc_diff + danger_penalty
    
    # # Strong advantage: > 75% of discs
    # if my_percentage > 0.75:
    #   # Bonus scales from 0 to +10 as percentage rises from 75% to 100%
    #   advantage_scale = (my_percentage - 0.75) / 0.25  # 0.0 to 1.0
    #   advantage_bonus = 10 * advantage_scale
    #   return disc_diff + advantage_bonus
    
    # Normal position
    return disc_diff
# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates

@register_agent("aurelia_agent")
class AureliaAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(AureliaAgent, self).__init__()
    self.name = "AureliaAgent"

  #So - this is shamelessly stolen from the greedy corners agent :P
  #I genuinely think this is pretty good
  #But then I saw Riv actually used this too, but weighted differently
  #So I figured I'd leave just this base version in
  #And we can play with the weighting when we meet?
  def evaluate_board(self, board, color, opponent):
              """
              Evaluate the board state based on multiple factors.

              Parameters:
              - board: 2D numpy array representing the game board.
              - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
              - player_score: Score of the current player.
              - opponent_score: Score of the opponent.

              Returns:
              - int: The evaluated score of the board.
              """
              # piece difference
              player_count = np.count_nonzero(board == color)
              opp_count = np.count_nonzero(board == opponent)
              score_diff = player_count - opp_count
              # corner control bonus
              n = board.shape[0]
              corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
              corner_bonus = sum(1 for (i, j) in corners if board[i, j] == color) * 5

              # penalize opponent mobility
              opp_moves = len(get_valid_moves(board, opponent))
              mobility_penalty = -opp_moves
              return score_diff + corner_bonus + mobility_penalty


  def a_b(self, board, depth, alpha, beta, max_player, color, opponent):
          """
          Algorithm done w/ help from geeks4geeks' page about minimax & a-b pruning (I am lazy)
          Except g4g uses minimax in the recursive step instead for like no reason when they could just call a_b again
          So I didn't do that
          """
          #get possibilities for current player
          moves = get_valid_moves(board, color if max_player else opponent)
          #we reached the bottom
          if depth < 1 or not moves:
              return self.evaluate_board(board, color, opponent)

          if max_player:
                  max_eval = float('-inf')
                  best_move = None
                  for move in moves:
                      new = deepcopy(board)
                      execute_move(new, move, color)
                      value = self.a_b(new, depth - 1, alpha, beta, False, color, opponent)
                      #print(value)
                      if isinstance(value, tuple):
                          value = value[0]
                      if value > max_eval:
                          max_eval = value
                          best_move = move
                      alpha = max(alpha, value)
                      if beta <= alpha:
                          break
                  return (max_eval, best_move)

          else:
                  min_eval = float('inf')
                  best_move = None
                  for move in moves:
                      new = deepcopy(board)
                      execute_move(new, move, opponent)
                      value = self.a_b(new, depth - 1, alpha, beta, True, color, opponent)
                      #print(value)
                      if isinstance(value, tuple):
                          value = value[0]

                      if value < min_eval:
                          min_eval = value
                          best_move = move
                      beta = min(beta, value)
                      if beta <= alpha:
                          break
                  return (min_eval, best_move)


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

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    move = None
    valid_moves = get_valid_moves(chess_board, player)
    move = self.a_b(chess_board, 2, float('-inf'), float('inf'), True, player, opponent)
    move = move[1]
    time_taken = time.time() - start_time
    #print("My AI's turn took ", time_taken, "seconds.")
    '''legal_moves = self.get_valid_unique_moves(chess_board, player)
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
            best_move = move'''
    return move


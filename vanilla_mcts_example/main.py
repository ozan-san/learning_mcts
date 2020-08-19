#!/usr/bin/env python
# coding: utf-8

# The same as before, but the winning value computation is slightly changed.
# Since we really care about the "possible win rate", that is, the times
# we've won out of all the games below, we will be updating the UCB computation,
# the backpropagation and the score calculation.
# See: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Principle_of_operation
# I also want to test whether the AI plays to a draw against itself. At least,
# this might be a step in the right direction regarding testing the algorithm.

import numpy as np
from copy import deepcopy
from math import sqrt
from math import log
from random import choice
import random

# Default variables to play with.
# Exploration coefficient (theoretically sqrt(2))
coeff = sqrt(2)
# The indicator variable for p1.
p1_indicator = 1  # X
# Indicator for p2: The negative of p1 (score calculations depend on this)
p2_indicator = -1 * p1_indicator # O


def equal_arrays(ar1, ar2):
    """
    A simple utility function for array equality.
    Args:
        ar1: Array 1.
        ar2: Array 2.

    Returns: Bool, indicating if the arrays are equal.

    """
    if len(ar1) != len(ar2):
        return False
    for i in range(len(ar1)):
        if ar1[i] != ar2[i]:
            return False
    return True


class NaughtsAndCrossesGame:
    """
    A class for the classical game, Naughts and Crosses.
    This implements the basic termination checks and scores for the game.
    """
    def __init__(self, board=None, player=1):
        """

        Args:
            board: List[List[int]]: The board as of the game start. Initially empty.
            player: The player to start.
        """
        if board is None:
            self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        else:
            self.board = board
        self.player = player

    def is_terminal(self):
        """

        Returns: Bool, indicating if the current game state is terminal or not.

        """
        x_wins = p1_indicator
        o_wins = p2_indicator
        x_arr = [x_wins, x_wins, x_wins]
        o_arr = [o_wins, o_wins, o_wins]

        # Check if all of the board is full or not
        not_found_empty = True
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    not_found_empty = False
                    break
        if not_found_empty:
            return True
        # Check rows
        for i in range(3):
            if equal_arrays(self.board[i], x_arr) or equal_arrays(self.board[i], o_arr):
                return True
        # Check columns
        transposed_board = list(np.transpose(np.array(self.board)))

        for i in range(3):
            if equal_arrays(transposed_board[i], x_arr) or equal_arrays(transposed_board[i], o_arr):
                return True
        # Check diagonals
        if self.board[0][0] == self.board[1][1] and self.board[2][2] == self.board[1][1] and self.board[1][1] != 0:
            return True
        if self.board[0][2] == self.board[1][1] and self.board[2][0] == self.board[1][1] and self.board[1][1] != 0:
            return True
        return False

    def get_score(self):
        """
        Return the score of the current board. This needs to be called
        after the check for terminal state is performed.
        Returns: p1_indicator, p2_indicator or 0 (in case of a draw)

        """
        # Scores are in the format:
        # (x_wincount, o_wincount)
        # x_wins = (1, 0, 0)
        # o_wins = (0, 1, 0)
        # draw = (0, 0, 1)
        x_wins = p1_indicator
        o_wins = p2_indicator
        x_arr = [x_wins, x_wins, x_wins]
        o_arr = [o_wins, o_wins, o_wins]
        draw = 0

        # Check rows

        for i in range(3):
            if equal_arrays(self.board[i], x_arr):
                return x_wins
            elif equal_arrays(self.board[i], o_arr):
                return o_wins

        # Check columns
        transposed_board = list(np.transpose(np.array(self.board)))

        for i in range(3):
            if equal_arrays(transposed_board[i], x_arr):
                return x_wins
            elif equal_arrays(transposed_board[i], o_arr):
                return o_wins

        # Check diagonals
        if self.board[0][0] == self.board[1][1] and self.board[2][2] == self.board[1][1] and (
                self.board[1][1] in [x_wins, o_wins]):
            return self.board[1][1]

        if self.board[0][2] == self.board[1][1] and self.board[2][0] == self.board[1][1] and (
                self.board[1][1] in [x_wins, o_wins]):
            return self.board[1][1]
        return draw

    def all_actions(self):
        """

        Returns: List[(int, int)], each representing a possible move

        """
        if self.is_terminal():
            return []
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    actions.append((i, j))
        return actions

    def play_action(self, action):
        """

        Args:
            action: (i: int,j: int), representing an intent on putting self.player
            on board position (i, j)

        Returns:
            NaughtsAndCrossesGame, representing the new state after the move is
            performed.
        """
        def toggler(num):
            if num == p1_indicator:
                return p2_indicator
            return p1_indicator

        board = deepcopy(self.board)
        board[action[0]][action[1]] = self.player
        return NaughtsAndCrossesGame(board, toggler(self.player))

    def all_states(self):
        """

        Returns: List[NaughtsAndCrossesGame], each representing a new
        game state after a possible move is performed.

        """
        actions = self.all_actions()
        states = []
        for action in actions:
            states.append(self.play_action(action))
        return states

    def __repr__(self):
        """

        Returns: The string representation of the game state.

        """
        resultarray = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == p1_indicator:
                    resultarray[i][j] = 'X'
                elif self.board[i][j] == p2_indicator:
                    resultarray[i][j] = 'O'
        for i in range(3):
            resultarray[i] = str(resultarray[i])
        return f"Player {self.player}:\n" + "\n".join(resultarray)


class MCTSNode:

    def __init__(self, parent, state):
        """

        Args:
            parent: MCTSNode, parent node. None if the current node is the root.
            state: Game state for MCTS.
        """
        self.parent = parent
        self.state = state
        self.visitCount = 0
        self.results = {p1_indicator: 0, p2_indicator: 0, 0: 0}
        # self.score = 0
        self.children = []
        self._states_not_seen = self.state.all_states()

    def expansion_finished(self):
        """

        Returns: bool, indicating if the expansion is finished.

        This implementation of mcts expands the nodes one by one.
        the usual implementation of mcts would achieve the same,
        with the visitCounts being 0, but this approach seems to
        explore more uniformly, as it does not have to explore the
        first move in the line.

        """
        return len(self._states_not_seen) == 0

    def expand(self):
        """

        Returns: MCTSNode, containing a new board state,
        constructed from a new random selection from actions
        remaining for this state.

        """
        # Get a random index in range
        i = random.randrange(len(self._states_not_seen))
        # Swap it with the last element
        self._states_not_seen[i], self._states_not_seen[-1] = self._states_not_seen[-1], self._states_not_seen[i]
        # Pop from list
        state = self._states_not_seen.pop()
        # Append the new child node, and return it (from the last position)
        self.children.append(MCTSNode(parent=self, state=state))
        return self.children[-1]

    def descend(self):
        """
        The selection phase of MCTS.
        Returns: MCTSNode, the node to be rolled out (then, to be backed up from).

        """
        if not self.state.is_terminal():
            if not self.expansion_finished():
                return self.expand()
            else:
                return self.best_child().descend()
        return self

    def get_score(self):
        """

        Returns: Int, the number of wins - losses.

        """
        win = self.results[self.state.player * -1]
        loss = self.results[self.state.player]
        return win - loss

    def uct_children(self, constant=coeff):
        """

        Args:
            constant: The exploration constant c for UCT formula.

        Returns: List[Float], the UCT value for each child.

        """
        return [(child.get_score() / child.visitCount) +
                constant * sqrt(log(self.visitCount) / child.visitCount) for child in self.children]

    def best_child(self, constant=coeff):
        """
        The best child according to UCT with given constant
        Args:
            constant: The exploration constant c for UCT formula.

        Returns: MCTSNode, the best node in terms of UCT value.

        """
        return self.children[np.argmax(self.uct_children(constant=constant))]

    def random_play(self):
        """
        The rollout phase. Until the state is terminal,
        a random move is played, eventually resulting
        in a score.
        Returns: Int, indicating score.

        """
        state = self.state
        while not state.is_terminal():
            action = choice(state.all_actions())
            state = state.play_action(action)
        return state.get_score()

    def visit(self):
        """
        Increment visitCount by 1.
        """
        self.visitCount += 1

    def backpropagate(self, result):
        """
        Back-propagates the result up the tree.
        Args:
            result: The result from the rollout phase.

        """
        self.visit()
        self.results[result] = self.results[result] + 1
        if self.parent:
            self.parent.backpropagate(result)

    def __repr__(self):
        """
        Returns: String representation of the tree node.

        """
        return f"V:{self.visitCount}, res:{self.results}, \n" + str(self.state)


def MCTS(gamestate, limit=1000):
    """
    Monte-Carlo Tree Search Algorithm, with given iteration limit.
    Args:
        gamestate: the game state to start the search from.
        limit: the iteration limit for the algorithm.

    Returns:
        gamestate: The new game state with the most desirable outcome,
        according to MCTS. An action could be returned as well, but it is
        a trivial task, and for ease of understanding, it's not implemented
        here.

    """
    # Iteration limit of 10k
    root = MCTSNode(parent=None, state=gamestate)
    iteration_count = 0
    while iteration_count < limit:
        iteration_count += 1
        selection = root.descend()
        score = selection.random_play()
        selection.backpropagate(score)

    return root.best_child(constant=0)


def comp_vs_comp():
    """
    Computer-vs-Computer.
    Returns: node, containing a terminal state.

    """
    state = NaughtsAndCrossesGame()
    while not state.is_terminal():
        node = MCTS(gamestate=state)
        state = node.state
        # print(state)
    return node


def test(trials=1000):
    """
    Test function, using comp_vs_comp above.
    Tests the MCTS algorithm for given number of trials.
    Args:
        trials: The number of trials.
    """
    xwincount, owincount, drawcount = 0, 0, 0
    i = 0
    while True:
        node = comp_vs_comp()
        state = node.state
        if state.get_score() == 0:
            # print(i, ": Draw")
            drawcount += 1
        elif state.get_score() == 1:
            # print(i, ": X Wins")
            xwincount += 1
        else:
            # print(i, ": O Wins")
            owincount += 1
        print(f"x:{xwincount}, o:{owincount}, d:{drawcount} out of {i + 1} games played.")
        i += 1


if __name__ == "__main__":
    test(100)

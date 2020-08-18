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

coeff = sqrt(2) / 2.0


def add_score(prev_score, to_add):
    return prev_score[0] + to_add[0], prev_score[1] + to_add[1], prev_score[2] + to_add[2]


def equal_arrays(ar1, ar2):
    if len(ar1) != len(ar2):
        return False
    for i in range(len(ar1)):
        if ar1[i] != ar2[i]:
            return False
    return True


class NaughtsAndCrossesGame:
    def __init__(self, board=None, player=1):
        if board is None:
            self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        else:
            self.board = board
        self.player = player

    def is_terminal(self):
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
            if equal_arrays(self.board[i], [1, 1, 1]) or equal_arrays(self.board[i], [2, 2, 2]):
                return True
        # Check columns
        transposed_board = list(np.transpose(np.array(self.board)))

        for i in range(3):
            if equal_arrays(transposed_board[i], [1, 1, 1]) or equal_arrays(transposed_board[i], [2, 2, 2]):
                return True
        # Check diagonals
        if self.board[0][0] == self.board[1][1] and self.board[2][2] == self.board[1][1] and (
                self.board[1][1] in [1, 2]):
            return True
        if self.board[0][2] == self.board[1][1] and self.board[2][0] == self.board[1][1] and (
                self.board[1][1] in [1, 2]):
            return True
        return False

    def get_score(self):
        # Scores are in the format:
        # (x_wincount, o_wincount)
        x_wins = (1, 0, 0)
        o_wins = (0, 1, 0)
        draw = (0, 0, 1)

        # Check rows

        for i in range(3):
            if equal_arrays(self.board[i], [1, 1, 1]):
                return x_wins
            elif equal_arrays(self.board[i], [2, 2, 2]):
                return o_wins

        # Check columns
        transposed_board = list(np.transpose(np.array(self.board)))

        for i in range(3):
            if equal_arrays(transposed_board[i], [1, 1, 1]):
                return x_wins
            elif equal_arrays(transposed_board[i], [2, 2, 2]):
                return o_wins

        # Check diagonals
        if self.board[0][0] == self.board[1][1] and self.board[2][2] == self.board[1][1] and (
                self.board[1][1] in [1, 2]):
            if self.board[1][1] == 1:
                return x_wins
            else:
                return o_wins
        if self.board[0][2] == self.board[1][1] and self.board[2][0] == self.board[1][1] and (
                self.board[1][1] in [1, 2]):
            if self.board[1][1] == 1:
                return x_wins
            else:
                return o_wins
        return draw

    def all_actions(self):
        if self.is_terminal():
            return []
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    actions.append((i, j))
        return actions

    def play_action(self, action):
        def toggler(num):
            if num == 1:
                return 2
            return 1

        board = deepcopy(self.board)
        board[action[0]][action[1]] = self.player
        return NaughtsAndCrossesGame(board, toggler(self.player))

    def all_states(self):
        actions = self.all_actions()
        states = []
        for action in actions:
            states.append(self.play_action(action))
        return states

    def __repr__(self):
        resultarray = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 1:
                    resultarray[i][j] = 'X'
                elif self.board[i][j] == 2:
                    resultarray[i][j] = 'O'
        for i in range(3):
            resultarray[i] = str(resultarray[i])
        return f"Player {self.player}:\n" + "\n".join(resultarray)


class MCTSNode:
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.visitCount = 0
        self.score = (0, 0, 0)
        self.children = []

    def visit(self):
        self.visitCount += 1

    def set_children(self, children):
        self.children = children

    def increment_score(self, newscore):
        self.score = add_score(self.score, newscore)

    def __repr__(self):
        return f"V:{self.visitCount}, S:{self.score}, \n" + str(self.state)


def backpropagate(leaf, score):
    node = leaf
    while not (node is None):
        node.visit()
        node.increment_score(score)
        node = node.parent
    # We will only use the scores in the ucb calculation and this will be sufficient.


def uct(node):
    # Taken from: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
    # w_i / n_i + c * sqrt(ln(N_i)/n_i)
    if node.visitCount == 0:
        return float('inf')
    if node.state.player == 1:
        win_count = node.score[0]
    else:
        win_count = node.score[1]
    draw_count = node.score[2]
    # TODO: Remove this!
    full_count = node.score[0] + node.score[1] + node.score[2]
    return ((win_count + draw_count) / full_count) + coeff * sqrt(log(node.parent.visitCount) / full_count)


def randomPlay(node):
    tempState = deepcopy(node.state)
    while not tempState.is_terminal():
        try:
            tempState = choice(tempState.all_states())
        except IndexError:
            print("Really?")
            print(tempState)
    return tempState.get_score()


def expand(node):
    newStates = node.state.all_states()
    children = [MCTSNode(parent=node, state=s) for s in newStates]
    node.set_children(children)


def descend(node):
    tempNode = node
    while len(tempNode.children) > 0:
        tempNode = tempNode.children[np.argmax([uct(c) for c in tempNode.children])]
    return tempNode


def MCTS(gamestate, limit=2000):
    # Iteration limit of 10k
    node = MCTSNode(parent=None, state=gamestate)
    iteration_count = 0
    while iteration_count < limit:
        iteration_count += 1
        node_to_expand = descend(node)
        if not node_to_expand.state.is_terminal():
            expand(node_to_expand)
            to_explore = choice(node_to_expand.children)
            result = randomPlay(to_explore)
            backpropagate(to_explore, result)
        else:
            backpropagate(node_to_expand, node_to_expand.state.get_score())

    # noinspection PyTypeChecker
    return node.children[np.argmax([c.visitCount for c in node.children])]


def comp_vs_comp():
    state = NaughtsAndCrossesGame()
    while not state.is_terminal():
        node = MCTS(gamestate=state)
        state = node.state
    return node

def test(trials=1000):
    xwincount, owincount, drawcount = 0, 0, 0
    for i in range(trials):
        print(i)
        node = comp_vs_comp()
        state = node.state
        if state.get_score() == (0,0,1):
            drawcount += 1
        elif state.get_score() == (1,0,0):
            xwincount += 1
        else:
            owincount += 1
    print(f"xwincount = {xwincount}, owincount = {owincount}, drawcount = {drawcount} / {trials}")


if __name__ == "__main__":
    test(100)

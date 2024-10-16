import copy
from itertools import pairwise
import matplotlib.pyplot as plt
import numpy as np

class Agent():
    def __init__(self):
        self.start = {'L': {'m': 3, 'c': 3},
                      'R': {'m': 0, 'c': 0},
                      'boat':  'L'} 
        self.goal  = {'L': {'m': 0, 'c': 0},
                      'R': {'m': 3, 'c': 3},
                      'boat':  'R'} 
        self.moves = [{'m': 1, 'c': 0}, {'m': 2, 'c': 0}, {'m': 0, 'c': 1}, {'m': 0, 'c': 2}, {'m': 1, 'c': 1}]
        #self.moves = [{'m': i, 'c': j} for i in range(3) for j in range(3) if i + j > 0 and i + j <= 2]
        self.frontier = [[self.start]]

    def is_valid(self, state):
        return True

    def apply_move(self, state, move):
        return state

    def next_states(self, state):
        return [state]

    def is_goal(self, state):
        return True

    def bfs(self):
        yield self.frontier[0]

    def states_to_moves(self, path):
        return []

    def solve(self):
        path = next(self.bfs())
        if path:
            print(*path, sep = '\n')
            print(*self.states_to_moves(path), sep='\n')

if __name__ == "__main__":
    agent = Agent()
    agent.solve()

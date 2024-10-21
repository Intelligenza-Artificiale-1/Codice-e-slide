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
        for side in ['L','R']:
            if state[side]['m']> 0 and state[side]['m'] < state[side]['c']:
                return False
            for type in ['m','c']:
                if state[side][type] < 0:
                    return False
        return True

    def apply_move(self, state, move):
        new_state = copy.deepcopy(state)
        #new_state['boat'] = 'R' if state['boat'] == 'L' else 'L'
        if state['boat'] == 'L':
            new_state['boat'] = 'R'
        else:
            new_state['boat'] = 'L'

        new_state[state['boat']]['m'] -= move['m']
        new_state[state['boat']]['c'] -= move['c']

        new_state[new_state['boat']]['m'] += move['m']
        new_state[new_state['boat']]['c'] += move['c']

        return new_state

    def next_states(self, state):
        # return ( s for m in moves if self.is_valid(s:=self.apply_move(state, m)))
        states = []
        for m in self.moves:
            s = self.apply_move(state, m)
            if self.is_valid(s):
                states.append(s)
        return states

    def is_goal(self, state):
        return state == self.goal

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if self.is_goal(path[-1]):
                yield path
            else:
                self.frontier += [path+[s] for s in self.next_states(path[-1]) if s not in path]

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

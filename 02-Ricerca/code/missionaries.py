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
        for sponda in ('L','R'):
            if not all(0 <= state[sponda][p] <= 3 for p in ('c','m')):
                return False
            if state[sponda]['m']>0 and  state[sponda]['m'] < state[sponda]['c']:
                return False
        return True

    def apply_move(self, state, move):
        # move = {'m':X, 'c':Y}
        new_state = copy.deepcopy(state)
        src = state['boat']
        dst = 'R' if src == 'L' else 'L'
        new_state['boat'] = dst
        for key in move:
            new_state[src][key] -= move[key]
            new_state[dst][key] += move[key]
        return new_state

    def next_states(self, state):
        #return ( s for m in self.moves if self.is_valid(s:= self.apply_move(s,m)))
        for move in self.moves:
            new_state = self.apply_move(state, move)
            if self.is_valid(new_state):
                yield new_state

    def is_goal(self, state):
        return state == self.goal

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if self.is_goal(path[-1]):
                yield path
            #for state in self.next_states(path[-1]):
            #    self.frontier.append(path + [state])
            self.frontier += [ path+[state] for state in self.next_states(path[-1]) if state not in path]

    def states_to_moves(self, path):
        #for s1, s2 in zip(path, path[1:]):
        for i in range(len(path)-1):
            s1 = path[i]
            s2 = path[i+1]
            for m in self.moves:
                if self.apply_move(s1, m) == s2:
                    yield m
                    break


if __name__ == "__main__":
    agent = Agent()
    solution = next(agent.bfs(), ["No path found"])
    print(*solution, sep="\n")

    print("====== MOVES =======")
    print(*agent.states_to_moves(solution), sep='\n')

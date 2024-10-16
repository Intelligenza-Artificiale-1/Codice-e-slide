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
        self.moves = [{'m': i, 'c': j} for i in range(3) for j in range(3) if i + j > 0 and i + j <= 2]
        self.frontier = [[self.start]]

    def is_valid(self, state):
        for side in ['L', 'R']:
            if state[side]['m'] < 0 or state[side]['c'] < 0:
                return False
            if state[side]['m'] > 0 and state[side]['c'] > state[side]['m']:
                return False
        return True

    def apply_move(self, state, move):
        new_state = copy.deepcopy(state)
        new_state['boat'] = 'L' if state['boat'] == 'R' else 'R'
        for person in ['m', 'c']:
            new_state[state['boat']][person] -= move[person]
            new_state[new_state['boat']][person] += move[person]
        return new_state

    def next_states(self, state):
        next_states = []
        for move in self.moves:
            new_state = self.apply_move(state, move)
            if self.is_valid(new_state):
                next_states.append(new_state)
        return next_states
        # return [new_state for move in self.moves if self.is_valid(new_state := self.apply_move(state, move))]

    def is_goal(self, state):
        return state == self.goal

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if self.is_goal(path[-1]):
                yield path
            self.frontier.extend([path + [state] for state in self.next_states(path[-1]) if state not in path])

    def states_to_moves(self, path):
        for i,j in pairwise(path):
            for move in self.moves:
                if self.apply_move(i, move) == j:
                    yield str(move['m']) + ' missionaries and ' + str(move['c']) + ' cannibals from ' + i['boat'] + ' to ' + j['boat']
                    break

    def plot_states(self, path):
        offsets= {'L': -1, 'R': 1}
        coords = []
        for i, state in enumerate(path):
            for side in ['L', 'R']:
                offset = offsets[side]
                for person, person_type in [('m', 0), ('c', 1)]:
                    for _ in range(state[side][person]):
                        coords.append([offsets[side] + offset, -i, person_type])
                        offset += offsets[side]
        coords = np.array(coords)
        plt.scatter(*coords[coords[:, 2] == 0][:, :2].T, c='b', label='missionaries')
        plt.scatter(*coords[coords[:, 2] == 1][:, :2].T, c='r', label='cannibals')
        # add legend
        plt.legend(['missionaries', 'cannibals'], loc='lower left')
        plt.show()

    def solve(self):
        path = next(self.bfs())
        if path:
            print(*path, sep = '\n')
            print(*self.states_to_moves(path), sep='\n')
            self.plot_states(path)

if __name__ == "__main__":
    agent = Agent()
    agent.solve()

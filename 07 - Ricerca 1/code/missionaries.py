import copy

class Agent():
    def __init__(self):
        self.start = {'L': {'missionaries': 3, 'cannibals': 3},
                      'R': {'missionaries': 0, 'cannibals': 0},
                      'boat':  'L'} 
        self.goal  = {'L': {'missionaries': 0, 'cannibals': 0},
                      'R': {'missionaries': 3, 'cannibals': 3},
                      'boat':  'R'} 
        self.moves = [{'missionaries': i, 'cannibals': j} for i in range(3) for j in range(3) if i + j > 0 and i + j <= 2]
        self.frontier = [[self.start]]

    def is_valid(self, state):
        left_ok= state['L']['missionaries'] >= state['L']['cannibals'] or state['L']['missionaries'] == 0
        right_ok = state['R']['missionaries'] >= state['R']['cannibals'] or state['R']['missionaries'] == 0
        return left_ok and right_ok

    def apply_move(self, state, move):
        boat = state['boat']
        new_state = copy.deepcopy(state)
        new_state[boat]['missionaries'] -= move['missionaries']
        new_state[boat]['cannibals'] -= move['cannibals']

        new_state['boat'] = 'L' if boat == 'R' else 'R'
        new_state[new_state['boat']]['missionaries'] += move['missionaries']
        new_state[new_state['boat']]['cannibals'] += move['cannibals']

        return new_state

    def next_states(self, state):
        boat = state['boat']
        next_states = []
        for move in self.moves:
            enough_miss = state[boat]['missionaries'] >= move['missionaries']
            enough_cann = state[boat]['cannibals'] >= move['cannibals']
            if enough_miss and enough_cann:
                new_state = self.apply_move(state, move)
                if self.is_valid(new_state):
                    next_states.append(new_state)
        return next_states

    def is_goal(self, state):
        return state == self.goal

    def bfs(self):
        if len(self.frontier) == 0:
            return None
        path = self.frontier[0]
        self.frontier = self.frontier[1:]
        if self.is_goal(path[-1]):
            return path
        next_paths = [path + [state] for state in self.next_states(path[-1]) if state not in path]
        self.frontier += next_paths
        return self.bfs()

    def states_to_moves(self, path):
        moves = []
        for i in range(len(path) - 1):
            for move in self.moves:
                if self.apply_move(path[i], move) == path[i + 1]:
                    moves.append(move)
                    break
        #convert to text
        for i in range(len(moves)):
            moves[i] = str(moves[i]['missionaries']) + ' missionaries and ' + str(moves[i]['cannibals']) + ' cannibals from ' + path[i]['boat'] + ' to ' + path[i + 1]['boat']
        return moves

    def solve(self):
        path = self.bfs()
        if path:
            return self.states_to_moves(path)
        else:
            return None

if __name__ == "__main__":
    agent = Agent()
    print(*agent.solve(), sep = '\n')

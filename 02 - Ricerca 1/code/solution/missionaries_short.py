import copy

class Agent():
    def __init__(self, start):
        self.frontier = [[start]]

    def next_states(self, state):
        for move in [{'m': i, 'c': j} for i in range(state[state['boat']]['m']+1) for j in range(state[state['boat']]['c']+1) if 0 < i + j <= 2]:
            new_state = copy.deepcopy(state)
            new_state['boat'] = (boat := 1 - state['boat'])
            for person in ['m', 'c']:
                new_state[boat][person] += move[person]
                new_state[1-boat][person] -= move[person]
            if all( (state[side]['m'] == 0 or state[side]['c'] <= state[side]['m']) for side in [0,1]):
                yield new_state

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if all(x==0 for x in path[-1][0].values()):
                yield path
            self.frontier.extend([path + [state] for state in self.next_states(path[-1]) if state not in path])

if __name__ == "__main__":
    agent = Agent(start={0: {'m': 3, 'c': 3}, 1: {'m': 0, 'c': 0}, 'boat':0})
    print(*next(agent.bfs(), []), sep='\n')
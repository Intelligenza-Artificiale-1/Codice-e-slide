class Agent():
    def __init__(self, state=[-1,1,0,1,1]):
        self.frontier=[[state]]

    def apply_move(self, state, i, j):
        new_state = state.copy()
        new_state[i], new_state[j] = new_state[j], new_state[i]
        for k in range(i+1,j):
            new_state[k] *=-1
        return new_state

    def next_states(self, state):
        empty = [i for i,v in enumerate(state) if v == 0]
        full  = [j for j,v in enumerate(state) if v != 0]
        for i1 in empty:
            for j1 in full:
                i,j = sorted([i1,j1])
                if (abs(i-j) > 1) and (0 not in state[i+1:j]):
                    yield self.apply_move(state, i, j)

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if all(x!=1 for x in path[-1]):
                return path
            self.frontier.extend(path + [state] for state in self.next_states(path[-1]) if state not in path)


if __name__ == '__main__':
    a = Agent()
    print(*a.bfs(), sep='\n')
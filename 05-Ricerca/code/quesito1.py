class Agent:
    def __init__(self, start=[-1,1,0,1,1]):
        self.frontier = [[start]]

    def is_goal(self, state):
        return not (1 in state)

    def next_states(self, state):
        pairs = ((i,j) for i in range(len(state)) for j in range(i+2, len(state)))
        for i,j in pairs:
            if (abs(state[i]) + abs(state[j]) == 1) and \
                0 not in state[i+1:j]:
                ns = state.copy()
                ns[i], ns[j] = ns[j], ns[i]
                ns[i+1:j] = [ -x for x in ns[i+1:j]]
                yield ns

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if self.is_goal(path[-1]):
                yield path
            self.frontier.extend(path + [s] for s in self.next_states(path[-1]) if s not in path)

if __name__ == '__main__':
    a = Agent()
    sol = next(a.bfs(),[])
    print(*sol, sep='\n')
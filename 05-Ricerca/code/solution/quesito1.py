class Agent():
    def __init__(self, start=[-1,1,0,1,1]):
        self.frontier = [[start]]

    def next_states(self, state):
        empty = [i for i,j in enumerate(state) if j==0]
        for orig, dest in (sorted([i, j]) for i in range(len(state)) if i not in empty for j in empty):
                if len(state[orig+1:dest]) > 0 and state[orig+1:dest].count(0) == 0:
                    new_state = state.copy()
                    new_state[dest] , new_state[orig] = state[orig], state[dest]
                    new_state[orig+1:dest] = [-x for x in state[orig+1:dest]]
                    yield new_state

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if not any(x==1 for x in path[-1]):
                return path
            self.frontier.extend(path + [state] for state in self.next_states(path[-1]) if state not in path)

if __name__ == "__main__":
    agent = Agent()
    print(f"Solution: ",*agent.bfs(), sep="\n")
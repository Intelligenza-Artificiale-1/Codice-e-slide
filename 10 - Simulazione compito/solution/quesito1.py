#agent that solves the 8 queens puzzle through depth first search
class Agent():
    def __init__(self):
        self.state = []

    def next_states(self, state):
        #returns a list of all possible next states
        for i in range(8):
            if i not in state:
                yield state + [i]

    def is_valid(self, state):
        #checks if a state is valid
        for i in range(len(state)):
            for j in range(i+1, len(state)):
                if state[i] == state[j] or abs(state[i] - state[j]) == j - i:
                    return False
        return True

    def dfs(self):
        if len(self.state) == 8:
            yield self.state
        else:
            for state in self.next_states(self.state):
                if self.is_valid(state):
                    self.state = state
                    yield from self.dfs()


if __name__ == "__main__":
    agent = Agent()
    for state in agent.dfs():
        print(state)

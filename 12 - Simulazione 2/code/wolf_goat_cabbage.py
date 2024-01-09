#agent that solves the wold, goat, cabbage river crossing problem using a breadth first search
class Agent():
    def __init__(self):
        self.start = {"wolf": 1, "goat": 1, "cabbage": 1, "boat": 1}
        self.goal = {"wolf": 0, "goat": 0, "cabbage": 0, "boat": 0}
        self.frontier = [[self.start]]

    def valid_state(self, state):
        if state["wolf"] == state["goat"] and state["boat"] != state["wolf"]:
            return False
        elif state["goat"] == state["cabbage"] and state["boat"] != state["goat"]:
            return False
        else:
            return True

    def next_states(self, state):
        next_states = []
        for key in state:
            if state["boat"] == state[key]:
                next_state = state.copy()
                next_state["boat"] = 1 - next_state["boat"]
                if key != "boat":
                    next_state[key] = 1 - next_state[key]
                next_states.append(next_state)
        return [s for s in next_states if self.valid_state(s)]

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            state = path[-1]
            if state == self.goal:
                return path
            for next_state in self.next_states(state):
                if next_state not in path:
                    self.frontier.append(path + [next_state])

if __name__ == "__main__":
    agent = Agent()
    print(*agent.bfs(), sep="\n")

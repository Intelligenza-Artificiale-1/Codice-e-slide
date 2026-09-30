WOLF, GOAT, CABBAGE, BOAT = 0, 1, 2, 3

class Agent():
    def __init__(self):
        self.frontier = [[[1,1,1,1]]]

    def is_valid(self, state):
        return not (state[WOLF] == state[GOAT] != state[BOAT] or state[GOAT] == state[CABBAGE] != state[BOAT])

    def next_states(self, state):
        boat_side = [idx for idx, value in enumerate(state) if value == state[BOAT]]
        moves = [ {BOAT} | {item} for item in boat_side ]  # all possible moves
        yield from ( new_state for move in moves if self.is_valid(new_state :=
             [1 - val if idx in move else val for idx, val in enumerate(state)]))

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if sum(path[-1]) == 0:
                yield path
            self.frontier.extend([path + [state] for state in self.next_states(path[-1]) if state not in path])

if __name__ == "__main__":
    agent = Agent()
    print(*next(agent.bfs(), ["No path found"]), sep="\n")
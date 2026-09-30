class Agent:
    def __init__(self, jugs=(3, 5, 20), start= (0, 0, 20), target=4):
        self.jugs = jugs
        self.target = target
        self.frontier = [[start]]

    def next_states(self, state):
        def is_valid(state):
            return all(0 <= curr <= cap for curr, cap in zip(state, self.jugs))

        moves = ((i,j) for i in range(len(self.jugs)) for j in range(len(self.jugs)) if i != j)
        for src, dst in moves:
            amount = min(state[src], self.jugs[dst] - state[dst])
            new_state = list(state)
            new_state[src] -= amount
            new_state[dst] += amount
            if is_valid(new_state):
                yield tuple(new_state)

    def is_goal(self, path): #potrebbe stare inline nella BFS
        return self.target in path[-1]

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if self.is_goal(path):
                return path
            for s in self.next_states(path[-1]):
                if s not in path:
                    self.frontier.append(path + [s])

if __name__ == "__main__":
    agent = Agent()
    print(agent.bfs())
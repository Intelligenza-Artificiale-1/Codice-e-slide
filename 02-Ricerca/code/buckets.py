class Agent():
    def __init__(self,jugs=(3,5,20), start=(0,0,20), target=4):
        self.jugs= jugs
        self.target = target
        self.frontier = [[start]]

    def is_goal(self, path):
        return self.target in path[-1]

    def next_states(self, path):
        #pairs = [(src, dst) for src in range(len(state)) for dst in range(len(state)) if src != dst]
        state = path[-1]
        pairs = []
        for src in range(len(state)):
            for dst in range(len(state)):
                if src != dst:
                    pairs.append((src,dst))

        for src, dst in pairs:
            disponibile = state[src]
            residuo = self.jugs[dst] - state[dst]
            effettivo = min(disponibile, residuo)
            if effettivo != 0:
                new_state = list(state)
                new_state[src] -= effettivo
                new_state[dst] += effettivo
                yield tuple(new_state)

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if self.is_goal(path):
                yield path
            for state in self.next_states(path):
                self.frontier.append(path+[state])

if __name__ == "__main__":
    a = Agent()
    print(next(a.bfs(), "No path found"))
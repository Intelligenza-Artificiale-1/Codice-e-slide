class DFSAgent:
    def __init__(self, colors, adjacency):
        self.colors = colors
        self.adjacency = adjacency

    def valid_coloring(self, state):
        for node in state:
            for neighbor in self.adjacency[node]:
                if neighbor in state and state[node] == state[neighbor]:
                    return False
        return True

    def next_moves(self, state):
        for node in self.adjacency:
            if node not in state:
                for color in self.colors:
                    yield (node,color)

    def dfs(self, state={}):
        if len(state) == len(self.adjacency):
            yield state
        for node, color in self.next_moves(state):
            new_state = state.copy()
            new_state[node] = color
            if self.valid_coloring(new_state):
                yield from self.dfs(state=new_state)


if __name__ == "__main__":
    adjacency = {'A': ['B','G','F'],
                'B': ['A','C'],
                'C': ['B','D','F','E'],
                'D': ['C'],
                'E': ['C'],
                'F': ['A','C','G'],
                'G': ['A','F']}

    colors = ['red','green','blue','yellow']
    a = DFSAgent(colors, adjacency)

    for state in a.dfs():
        print(state)
        # ask user if they want to continue
        print("Would you like to continue? (Y/n)")
        answer = input()
        if answer == 'n':
            break

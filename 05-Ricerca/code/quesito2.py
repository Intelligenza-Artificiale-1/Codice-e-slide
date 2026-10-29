import copy

class Agent:
    def __init__(self, grid):
        self.condimenti = grid

    def is_valid(self, state):
        pairs = []
        slices = { x for row in state for x in row if x!=0 }
        for s in slices:
            coords = [(i,j) for i in range(len(state)) for j in range(len(state[0])) if state[i][j] == s ]
            pair = []
            for i,j in coords:
                pair.append(self.condimenti[i][j])
            pairs.append(tuple(sorted(pair)))
        return len(pairs) == len(set(pairs))
        
    def next_states(self, state):
        row = min((i for i in range(len(state)) if 0 in state[i]), default=None)
        if row is not None:
            col = min(j for j in range(len(state[row])) if state[row][j]==0)
            n   = max( x for row in state for x in row ) +1
            for neigh in [(0,1), (1,0)]:
                new_state = copy.deepcopy(state)
                new_state[row][col] = n
                nrow, ncol = row+neigh[0], col+neigh[1]
                if 0 <= nrow < len(state) and 0 <= ncol < len(state[0]) and state[nrow][ncol] == 0:
                    new_state[nrow][ncol] = n
                    if self.is_valid(new_state):
                        yield new_state

    def dfs(self, state=None):
        if state==None:
            state = [ [0 for _ in self.condimenti[0]] for _ in self.condimenti]
        if all(all(x!=0 for x in row) for row in state):
            yield state
        for s in self.next_states(state):
            yield from self.dfs(s)


if __name__ == "__main__":
    grid = [
        [0,1,4,0,5,3,2],
        [3,3,2,4,6,0,6],
        [6,5,4,3,1,2,4],
        [2,0,5,5,1,6,3],
        [5,4,1,2,0,6,6],
        [0,1,2,3,4,5,1],
    ]
    agent = Agent(grid=grid)
    for solution in agent.dfs():
        print(f"Solution: ",*solution, sep="\n")
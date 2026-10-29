import copy
class Agent():
    def __init__(self, grid:list):
        self.grid = grid

    def next_pairs(self, state:list):
        if (row :=min((i for i, r in enumerate(state) if min(r) < 'A'), default=None)) is not None:
            col = state[row].index(chr(ord('A')-1))
            yield from ( ((row, col), (row + dr, col + dc))
                for dr, dc in [(0,1), (1,0)]
                if row + dr < len(state) and col + dc < len(state[0])
                and state[row + dr][col + dc] < 'A')

    def color_pair(self, state:list, pair:tuple):
        new_state = copy.deepcopy(state)
        new = chr(1+ord(max(max (row) for row in new_state)))
        new_state[pair[0][0]][pair[0][1]] = new
        new_state[pair[1][0]][pair[1][1]] = new
        return new_state

    def is_valid(self, state:list):
        colors = { c for row in state for c in row if c >= 'A' }
        values = { frozenset({self.grid[i][j] for i, row in enumerate(state) for j, c2 in enumerate(row) if c2 == c}) for c in colors }
        return len(values) == len(colors) 

    def dfs(self, state=None):
        if state is None:
            state =  [[chr(ord('A')-1)] * len(row) for row in grid]
        if all(c >= 'A' for row in state for c in row):
            yield state
        for pair in self.next_pairs(state):
            if self.is_valid(new_state := self.color_pair(state, pair)):
                yield from self.dfs(new_state)
        

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
class Agent():
    def __init__(self, start=(0,0), size=4, p=4, fuel=[(0,3),(1,1),(3,1)]):
        self.start = start
        self.size = size
        self.fuel = set(fuel) | set([self.start])
        self.p = p

    def valid_path(self, path):
        return all(all(0 <= e < self.size for e in c) for c in path) and \
            len(path)==len(set(path)) and \
            all(set(path[i:i+self.p]) & self.fuel for i in range(0, len(path) - self.p))
            
    def next_paths(self, path):
        neighbors = [ (path[-1][0] + dx,path[-1][1] + dy) for dx, dy in ((1,0), (-1,0), (0,1), (0,-1))]
        yield from ( new for step in neighbors if  self.valid_path(new := path + [step]) )

    def dfs(self, path=None):
        if path is None:
            path = [self.start]
        if len(path) == self.size ** 2:
            yield path
        for next_path in self.next_paths(path):
            yield from self.dfs(next_path)

if __name__ == "__main__":
    agent = Agent()
    print(f"Solution: ",*agent.dfs(), sep="\n")
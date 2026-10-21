import argparse
from time import sleep
import tracemalloc
import timeit
#Code to solve the 8 game problem using BFS, greedy, and A* search


def state_to_str(state):
    # Converte lo stato in una stringa 3x3
    return "\n".join([" ".join([str(state[3 * i + j]) for j in range(3)]) for i in range(3)]).replace("0", " ")

class Agent():
    def __init__(self, start, goal=[1, 2, 3, 4, 5, 6, 7, 8, 0]):
        self.generated_states = 0
        self.start = start
        self.goal = goal
        # Lo spazio vuoto può essere spostato in 4 direzioni
        self.moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def move(self, state, move):
        # Calcola la nuova posizione dello spazio vuoto
        empty = state.index(0)
        new_empty = empty + move[0] + 3 * move[1]
        # Controlla se la nuova posizione è valida
        if (new_empty < 0 or new_empty > 8) or \
                (empty % 3 == 0 and move[0] == -1) or \
                (empty % 3 == 2 and move[0] == 1):
            return None
        new_state = state.copy()
        new_state[empty], new_state[new_empty] = new_state[new_empty], new_state[empty]
        return new_state

    def next_paths(self, path):
        # Calcola le prossime mosse possibili
        for move in self.moves:
            if (new := self.move(path[-1], move)) and new not in path:
                self.generated_states += 1
                yield path + [new]

    def search(self):
        raise NotImplementedError

class AgentBFS(Agent):
    def __init__(self, start, goal=[1, 2, 3, 4, 5, 6, 7, 8, 0]):
        super().__init__(start, goal)
        self.frontier = [[start]]

    def __bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if path[-1] == self.goal:
                yield path
            self.frontier += [next_path for next_path in self.next_paths(path)]

    def search(self):
        return self.__bfs()

class AgentDFS(Agent):
    def __init__(self, start, goal=[1, 2, 3, 4, 5, 6, 7, 8, 0]):
        super().__init__(start, goal=goal)
        self.state = [start]

    def __dfs(self, depth=20):
        if depth>0:
            if self.state[-1] == self.goal:
                yield self.state
            for next_path in self.next_paths(self.state):
                self.state = next_path
                yield from self.__dfs(depth=depth - 1)

    def search(self, depth=20):
        return self.__dfs(depth=depth)

class AgentIDS(AgentDFS):
    def __init__(self, start, goal=[1, 2, 3, 4, 5, 6, 7, 8, 0]):
        super().__init__(start, goal=goal)

    def __ids(self):
        depth = 0
        while True:
            self.state = [self.start]
            yield from super().search(depth=depth)
            depth += 1

    def search(self):
        return self.__ids()

class AgentGreedy(AgentBFS):
    def __init__(self, start, goal=[1, 2, 3, 4, 5, 6, 7, 8, 0]):
        super().__init__(start, goal)

    def heuristic(self, path):
        # Calcola il numero di tessere fuori posto
        return sum([1 if i != j else 0 for i,j in zip(path[-1], self.goal)])

    def __greedy(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if path[-1] == self.goal:
                yield path
            new_states = [next_path for next_path in self.next_paths(path)]
            self.frontier += new_states
            self.frontier.sort(key=lambda path: self.heuristic(path))

    def search(self):
        return self.__greedy()

class AgentAStar(AgentGreedy):
    def __init__(self, start, goal=[1, 2, 3, 4, 5, 6, 7, 8, 0]):
        super().__init__(start, goal)

    def heuristic(self, path):
        return super().heuristic(path) + len(path)

def get_solution(start, method="bfs", interactive=False):
    if method == "bfs":
        a = AgentBFS(start)
    elif method == "dfs":
        a = AgentDFS(start)
    elif method == "ids":
        a = AgentIDS(start)
    elif method == "greedy":
        a = AgentGreedy(start)
    elif method == "astar":
        a = AgentAStar(start)
    else:
        raise Exception("Invalid method")
    s = a.search()
    sol = next(s,None)
    return  sol, a.generated_states
       
if __name__ == "__main__":
    start = [ 2, 4, 3, 7, 1, 0, 8, 6, 5 ]
    #start = [ 2, 4, 3, 7, 0, 1, 8, 6, 5 ]
    parser = argparse.ArgumentParser(description="Solve the 8 game problem")
    parser.add_argument("-m", "--method", choices=["bfs", "dfs", "ids", "greedy", "astar"], default="bfs", help="Search method")
    parser.add_argument("-p", "--print", action="store_true", help="Print the solution")
    args = parser.parse_args()
    #evaluate the time and memory usage of the solution
    time = timeit.timeit(lambda: get_solution(start, method=args.method), number=10)
    tracemalloc.start()
    solution, generated_states = get_solution(start, method=args.method)
    _, memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Generated states: {generated_states:3d}, time: {time:.4f}, memory: {memory/1024:.0f} KiB, solution depth: {len(solution)}")
    if args.print:
        for state in solution:
            print(state_to_str(state), end="\r\033[2A", flush=True)
            sleep(0.5)
import argparse
import tracemalloc
import timeit
#Code to solve the 8 game problem using BFS, greedy, and A* search

class Agent():
    def __init__(self, start, goal=[1, 2, 3, 4, 5, 6, 7, 8, 0]):
        self.generated_states = 0
        self.start = start
        self.goal = goal
        # Lo spazio vuoto può essere spostato in 4 direzioni
        self.moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.frontier = [[start]]

    def next_paths(self, path):
        # Calcola le prossime mosse possibili
        state = path[-1]
        for move in self.moves:
            new_state = self.move(state, move)
            if new_state and new_state not in path:
                self.generated_states += 1
                yield path + [new_state]

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

    def bfs(self):
        if self.frontier:
            path = self.frontier.pop(0)
            if path[-1] == self.goal:
                yield path
            new_states = [next_path for next_path in self.next_paths(path)]
            self.frontier += new_states
            yield from self.bfs()

    def dfs(self, depth=0):
        if depth:
            path = self.frontier.pop(0)
            if path[-1] == self.goal:
                yield path
            for next_path in self.next_paths(path):
                self.frontier = [next_path]
                yield from self.dfs(depth=depth - 1)

    def ids(self):
        depth = 0
        while True:
            self.frontier = [[self.start]]
            yield from self.dfs(depth=depth)
            depth += 1

    def heuristic(self, state):
        # Calcola il numero di tessere fuori posto
        return sum([1 if state[i] != self.goal[i] else 0 for i in range(len(state))])

    def greedy(self):
        if self.frontier:
            path = self.frontier.pop(0)
            if path[-1] == self.goal:
                yield path
            new_states = [next_path for next_path in self.next_paths(path)]
            self.frontier += new_states
            self.frontier.sort(key=lambda path: self.heuristic(path[-1]))
            yield from self.greedy()

    def astar(self):
        if self.frontier != []:
            path = self.frontier.pop(0)
            if path[-1] == self.goal:
                yield path
            new_states = [next_path for next_path in self.next_paths(path)]
            self.frontier += new_states
            self.frontier.sort(key=lambda path: self.heuristic(path[-1]) + len(path))
            yield from self.astar()

def state_to_str(state):
    # Converte lo stato in una stringa 3x3
    return "\n".join([" ".join([str(state[3 * i + j]) for j in range(3)]) for i in range(3)])


def get_solution(start, method="bfs", interactive=False):
    a = Agent(start)
    if method == "bfs":
        s=a.bfs()
    elif method == "ids":
        s=a.ids()
    elif method == "greedy":
        s=a.greedy()
    elif method == "astar":
        s=a.astar()
    else:
        raise Exception("Invalid method")
    if interactive:
        for path in s:
            print(*[state_to_str(state) for state in path], sep="\n\n")
            print("Generate more? [Y/n]")
            if input() == "n":
                break
    else:
        sol = next(s)
        return  sol, a.generated_states
       
if __name__ == "__main__":
    start = [ 2, 4, 3, 7, 1, 5, 8, 0, 6 ]
    parser = argparse.ArgumentParser(description="Solve the 8 game problem")
    parser.add_argument("-m", "--method", choices=["bfs", "ids", "greedy", "astar"], default="bfs", help="Search method")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    if not args.interactive:
        time = timeit.timeit(lambda: get_solution(start, method=args.method), number=100)
        tracemalloc.start()
        solution, generated_states = get_solution(start, method=args.method)
        _, memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Generated states: {generated_states:3d}, time: {time:.4f}, memory: {memory/1024:.0f} KiB")
    else:
        get_solution(start, method=args.method, interactive=args.interactive)


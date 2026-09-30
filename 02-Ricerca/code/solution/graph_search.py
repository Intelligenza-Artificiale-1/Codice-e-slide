#The graph is represented as a dictionary of dictionaries
#The keys of the outer dictionary are the nodes of the graph
#The values of the outer dictionary are dictionaries that map a node to its neighbors

graph = {'A': ['B', 'C'],
         'B': ['A', 'D', 'E'],
         'C': ['A', 'F'],
		 'D': ['B', 'G'],
         'E': ['B', 'G'],
         'F': ['C', 'G'],
         'G': ['D', 'E', 'F']}

# Each arc is represented as a tuple (node1, node2)
graph2 = [('A', 'B'), ('A', 'C'),
          ('B', 'A'), 
          ('B', 'D'), ('B', 'E'),
          ('C', 'A'), ('C', 'F'),
          ('D', 'B'), ('D', 'G'),
          ('E', 'B'), ('E', 'G'),
          ('F', 'C'), ('F', 'G'),
          ('G', 'D'), ('G', 'E'), ('G', 'F')]

class Agent():
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.goal  = goal
        self.frontier = [[start]]

    def next_states(self, path):
        return self.graph[path[-1]]

    def is_goal(self, state):
        return state == self.goal

    def bfs(self):
        if len(self.frontier) == 0:
            return None
        path = self.frontier.pop(0)
        if self.is_goal(path[-1]):
            yield path
        # NOTA BENE: se avessimo un "else" qui, una volta trovato il goal,
        # non verrebbero più cercati altri percorsi nonostante l'utilizzo
        # del "yield" (che permette di continuare la ricerca)
        next_paths = []
        for state in self.next_states(path):
            if state not in path:
                next_paths.append(path + [state])
        self.frontier += next_paths
        yield from self.bfs() 

class Agent2(Agent):
    # graph as list of arcs, bfs iterative instead of recursive
    def __init__(self, graph, start, goal):
        super().__init__(graph, start, goal)

    def next_states(self, path):
        yield from (to for frm, to in self.graph if frm == path[-1])

    def bfs(self):
        while self.frontier:
            if self.is_goal((path:=self.frontier.pop(0))[-1]):
                yield path
            self.frontier.extend([path + [state] for state in self.next_states(path) if state not in path])


if __name__ == "__main__":
    edgelist = False
    if edgelist:
        a = Agent2(graph2, 'A', 'D')
    else:
        a = Agent(graph, 'A', 'D')
    print(next(a.bfs(), "No path found"))
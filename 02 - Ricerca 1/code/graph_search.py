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
graph2 = []

class Agent():
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.goal  = goal
        self.frontier = [[start]]

    def next_states(self, path):
        pass

    def is_goal(self, state):
        pass

    def bfs(self):
        #...
        yield from self.bfs() 

class Agent2(Agent):
    # graph as list of arcs, bfs iterative instead of recursive
    def __init__(self, graph, start, goal):
        super().__init__(graph, start, goal)

    def next_states(self, path):
        pass

    def bfs(self):
        pass



if __name__ == "__main__":
    edgelist = False
    if edgelist:
        a = Agent2(graph2, 'A', 'D')
    else:
        a = Agent(graph, 'A', 'D')
    print(next(a.bfs(), "No path found"))
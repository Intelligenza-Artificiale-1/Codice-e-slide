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
graph2 =[('A','B') ,('A','C'),
         ('B','A'), ('B','D'), ('B','E'),
         ('C', 'A'), ('C','F'),
		 ('D', 'B'), ('D','G'),
         ('E', 'B'), ('E','G'),
         ('F', 'C'), ('F','G'),
         ('G', 'D'), ('G','E'), ('G','F')]

class Agent():
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.goal  = goal
        self.frontier = [[start]]

    def next_states(self, path):
        #return [ n for n in graph[path[-1]] if n not in path ]
        states = []
        for n in graph[path[-1]]:
            if n not in path:
                states.append(n)
        return states

    def is_goal(self, state):
        return state == self.goal

    def bfs(self):
        if self.frontier == []:
            yield []
        else:
            path = self.frontier.pop(0)
            if self.is_goal(path[-1]):
                yield path
            # self.frontier += [path+[n] for n in self.next_states(path)]
            for n in self.next_states(path):
                self.frontier.append(path+[n])
            yield from self.bfs()

class Agent2(Agent):
    # graph as list of arcs, bfs iterative instead of recursive
    def __init__(self, graph, start, goal):
        super().__init__(graph, start, goal)

    def next_states(self, path):
        # return ( j for i,j in self.graph if i == path[-1] and j not in path )
        states = []
        for i,j in self.graph:
            if i == path[-1] and j not in path:
                states.append(j)
        return states


    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if self.is_goal(path[-1]):
                yield path
            else:
                self.frontier += [path+[n] for n in self.next_states(path)]



if __name__ == "__main__":
    edgelist = True
    if edgelist:
        a = Agent2(graph2, 'A', 'G')
    else:
        a = Agent(graph, 'A', 'G')
    min_length = float('inf')
    for path in a.bfs():
        if len(path) <= min_length:
            min_length = len(path)
            print(path)
        else:
            break
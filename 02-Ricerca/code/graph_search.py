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
        curr_state = path[-1]
        neighbors = self.graph[curr_state]
        return (n for n in neighbors if n not in path)

    def is_goal(self, state):
        return state == self.goal

    def bfs(self):
        if len(self.frontier)==0:
            return None
        path = self.frontier.pop(0)
        if self.is_goal(path[-1]):
            yield path
        for state in self.next_states(path):
            self.frontier.append(path+[state])
        yield from self.bfs() 

class Agent2(Agent):
    # graph as list of arcs, bfs iterative instead of recursive
    def __init__(self, graph, start, goal):
        super().__init__(graph, start, goal)

    def next_states(self, path):
        curr_state = path[-1]
        return ( y for x,y in self.graph if x==curr_state and y not in path )
        """nodi_raggiungibili = []
        for x,y in self.graph:
            if x == curr_state:
                nodi_raggiungibili.append(y)
        return nodi_raggiungibili"""

    def bfs(self):
        while self.frontier:
            path = self.frontier.pop(0)
            if self.is_goal(path[-1]):
                yield path
            #self.frontier.extend( path + [state] for state in self.next_states(path))
            for state in self.next_states(path):
                self.frontier.append(path+[state])


if __name__ == "__main__":
    edgelist = True
    if edgelist:
        a = Agent2(graph2, 'A', 'D')
    else:
        a = Agent(graph, 'A', 'D')
    print(next(a.bfs(), "No path found"))
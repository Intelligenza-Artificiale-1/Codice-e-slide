#Python code for problem solving using graph search algorithms (DFS,BFS)

import sys
from collections import deque
import numpy as np

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
        path = self.frontier[0]
        self.frontier = self.frontier[1:]
        if self.is_goal(path[-1]):
            yield path
        next_paths = [path + [state] for state in self.next_states(path) if state not in path]
        self.frontier += next_paths
        yield from self.bfs() 

if __name__ == "__main__":
    a = Agent(graph, 'A', 'B')
    min_length = float('inf')
    for path in a.bfs():
        if len(path) <= min_length:
            min_length = len(path)
        print(path)

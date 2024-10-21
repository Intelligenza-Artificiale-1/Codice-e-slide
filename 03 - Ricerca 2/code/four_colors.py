class DFSAgent:
    def __init__(self, colors, adjacency):
        self.colors = colors
        self.adjacency = adjacency
        self.state = {}

    def valid_coloring(self, state, new_node=None):
        return True

    def next_states(self, state):
        yield state

    def dfs(self):
        yield self.state

from itertools import chain
class DFSAgentIterative(DFSAgent):
    def __init__(self, colors, adjacency):
        super().__init__(colors, adjacency)
        self.frontier = (s for s in [{}])

    def dfs(self):
        while (state := next(self.frontier),None)!= None:
            if len(state) == len(self.adjacency):
                yield state
            self.frontier = chain.from_iterable([self.next_states(state), self.frontier])

class DFSAgentIterative2(DFSAgent):
    def __init__(self, colors, adjacency):
        super().__init__(colors, adjacency)
        self.frontier = [{}, None]

    def dfs(self):
        while (state := self.frontier.pop(0))!= None:
            if len(state) == len(self.adjacency):
                print(self.generated_states)
                yield state
            self.frontier[0:0] = list(self.next_states(state)) 

class DFSAgentOptim(DFSAgent):
    def __init__(self, colors, adjacency, mrw=True, lcv=True, fwc=True):
        super().__init__(colors, adjacency)
        self.fwc, self.lcv, self.mrw = fwc, lcv, mrw

    def legal_values(self, state, node):
        return len(self.colors) - len({state[neighbor] for neighbor in self.adjacency[node] if neighbor in state})

    def forward_checking(self, state):
        return True

    def MCV_sort(self, state):
        return [n for n in self.adjacency if n not in state][0]

    def removed_neighbors_colors(self, state, node, color):
        neighbors = [ n for n in self.adjacency[node] if n not in state]
        return 0

    def LCV_sort(self, state, node):
        return self.colors

    def next_states(self, state):
        node = self.MCV_sort(state)
        return ( new_state 
                for color in self.LCV_sort(state, node)
                if self.valid_coloring(new_state := {**state, node: color}, new_node=node) and self.forward_checking(new_state))

if __name__ == "__main__":
    adjacency = {'A': ['B','G','F'],
                'B': ['A','C'],
                'C': ['B','D','F','E'],
                'D': ['C'],
                'E': ['C'],
                'F': ['A','C','G'],
                'G': ['A','F']}

    #make a much larger graph
    adjacency2 = {
        1: [2, 4, 11, 13],
        2: [1, 3, 10, 12],
        3: [2, 6, 8, 11],
        4: [1, 5, 8, 10, 14],
        5: [4, 7, 9, 13],
        6: [3, 7, 9, 12],
        7: [5, 6, 8, 9, 14],
        8: [3, 4, 7, 9, 12, 13],
        9: [5, 6, 7, 8, 14],
        10: [2, 4],
        11: [1, 3],
        12: [2, 6, 8],
        13: [1, 5, 8],
        14: [4, 7, 9],
    }

    adjacency3 = {
        1: [2, 4, 11, 13],
        2: [1, 3, 10, 12],
        3: [2, 6, 8, 11, 15, 17],
        4: [1, 5, 8, 10, 14, 17],
        5: [4, 7, 9, 13, 16, 18],
        6: [3, 7, 9, 12, 16, 18],
        7: [5, 6, 8, 9, 14, 15, 17, 18],
        8: [3, 4, 7, 9, 12, 13, 16, 18],
        9: [5, 6, 7, 8, 14, 15, 16, 17],
        10: [2, 4, 20, 22, 29],
        11: [1, 3, 19, 21, 29],
        12: [2, 6, 8, 20, 24, 26, 29],
        13: [1, 5, 8, 19, 23, 26, 29],
        14: [4, 7, 9, 22, 25, 27, 29],
        15: [3, 7, 9, 21, 25, 27, 29],
        16: [5, 6, 8, 9, 23, 24, 26, 27, 29],
        17: [3, 4, 7, 9, 21, 22, 25, 27, 29],
        18: [5, 6, 7, 8, 23, 24, 25, 26, 29],
        19: [11, 13, 28, 30],
        20: [10, 12, 28, 30],
        21: [11, 15, 17, 28, 30],
        22: [10, 14, 17, 28, 30],
        23: [13, 16, 18, 28, 30],
        24: [12, 16, 18, 28, 30],
        25: [14, 15, 17, 18, 28, 30],
        26: [12, 13, 16, 18, 28, 30],
        27: [14, 15, 16, 17, 28, 30],
        28: [19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30],
        29: [10, 11, 12, 13, 14, 15, 16, 17, 18, 28, 30],
        30: [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    }

    colors = ['red','green','blue','yellow']
    a = DFSAgent(colors, adjacency)

    for state in a.dfs():
        print(state)
        # ask user if they want to continue
        print("Would you like to continue? (Y/n)")
        answer = 'n' #input()
        if answer == 'n':
            break

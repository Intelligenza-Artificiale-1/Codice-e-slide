class Agent():
    """Depth-first Search Agent instantiated with a modified 
    version of the Knapsack problem."""

    def __init__(self, W=0, V=0, items={}):
        """Initialize the agent with a minimum value and a list of items."""
        self.W, self.V = W, V
        self.items = list(items.values())
        self.state = []

    def next_states(self, state):
        """Generate the next possible states from the current state."""
        yield from [state + [b] for b in [True, False]]
        
    def is_valid(self, state):
        """Check if the current state is valid."""
        return len(state) <= len(self.items) and sum(self.items[idx][1] for idx, val in enumerate(state) if val) <= self.W

    def search(self):
        """Search for the best combination of items to meet the minimum value."""
        if sum(self.items[idx][0] for idx, val in enumerate(self.state) if val) > self.V and len(self.state) == len(self.items):
            yield self.state
        for new_state in self.next_states(self.state):
            if self.is_valid(new_state):
                self.state = new_state
                yield from self.search()
        

if __name__ == "__main__":
    items = {   'bottle': (1, 600), 
                'binoculars': (1, 900),
                'map': (2, 150), 
                'compass': (1, 200),
                'food': (2, 500),
                'clothes': (2, 400),
                'tent': (3, 1000),
                'sleeping_bag': (2, 800),
                'stove': (2, 700)
            }
    agent = Agent(W=1400, V=6, items=items)
    for state in agent.search():
        print([list(items.keys())[i] for i, val in enumerate(state) if val])
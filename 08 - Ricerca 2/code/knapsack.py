class UCSAgent():
    """Uniform Cost Search Agent instantiated with a modified 
    version of the Knapsack problem."""

    def __init__(self, minimum_value, items):
        """Initialize the agent with a minimum value and a list of items."""
        self.minimum_value = minimum_value
        self.items = items
        self.frontier = [set()]

    def state_value(self, state):
        """Return the value of the state."""
        return sum([self.items[i][0] for i in state])

    def state_weight(self, state):
        """Return the weight of the state."""
        return sum([self.items[i][1] for i in state])

    def search(self):
        """Search for the best combination of items to meet the minimum value."""
        if self.frontier:
            node = self.frontier.pop(0)
            if self.state_value(node) >= self.minimum_value:
                yield node
            for item in self.items.keys():
                if item not in node:
                    new_node = node | {item}
                    if new_node not in self.frontier:
                        self.frontier.append(new_node)
            self.frontier.sort(key=lambda x: self.state_weight(x))
            yield from self.search()

if __name__ == "__main__":
    items = {
                'bottle': (1, 600), 
                'binoculars': (2, 900),
                'map': (2, 150), 
                'compass': (1, 200)
            }
    agent = UCSAgent(4, items)
    for state in agent.search():
        print(f"Value: {agent.state_value(state)}", f"Weight: {agent.state_weight(state)}", f"Items: {state}", sep="\n")
        print("Would you like to continue searching? (Y/n)")
        if input().lower() == 'n':
            break

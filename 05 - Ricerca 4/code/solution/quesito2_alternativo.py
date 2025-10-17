import copy
class Agent():
    def __init__(self, grid):
        self.grid = grid
        self.state = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        self.pairs = set()

    def next_states(self, state, pairs):
        #determina la prima cella non assegnata ad una fetta
        for i, row in enumerate(state):
            if 0 in row:
                j = row.index(0)
                break
        else:
            return  #tutte le celle sono assegnate
                
        #determina il numero della fetta successiva
        #alternativamente potevamo contare la lunghezza di pairs e sommare 1
        counter = max(max(row) for row in state) +1
        for dx, dy in [(0,1),(1,0)]:
            #guardiamo le due direzioni possibili: destra e giù
            i1, j1 = i+dx, j+dy
            #se la cella è valida e non assegnata, creiamo un nuovo stato
            if (0<=i1<len(state)) and (0<=j1<len(state[0])) and (state[i1][j1]==0):
                new_state, new_pairs = copy.deepcopy(state), copy.deepcopy(pairs)
                # Annotiamo nelle due celle con il numero della fetta
                new_state[i][j], new_state[i1][j1] = counter, counter
                # Aggiungiamo la combinazione di frutti usata in questa fetta
                new_pairs.add(tuple(sorted((self.grid[i][j], self.grid[i1][j1]))))
                # Verifichiamo che la nuova combinazione di frutti non sia già stata usata
                # questo controllo non può essere fatto fuori da questo metodo.
                # Altrimenti rn fase di backtracking perderemmo l'informazione 
                # su quali combinazioni sono già state usate
                # (Era il bug che avevamo riscontrato a lezione)
                if len(new_pairs) > len(pairs):
                    yield new_state, new_pairs
                # In alternativa
                # new_combination = tuple(sorted((self.grid[i][j], self.grid[i1][j1])))
                # if new_combination not in pairs:
                #     yield new_state, new_pairs

    def dfs(self):
        # condizione di terminazione: tutte le celle sono assegnate
        if all(all(x != 0 for x in row) for row in self.state):
            yield self.state
        for next_state in self.next_states(self.state, self.pairs):
            self.state, self.pairs = next_state
            yield from self.dfs()
            


if __name__ == "__main__":
    grid = [
        [0,1,4,0,5,3,2],
        [3,3,2,4,6,0,6],
        [6,5,4,3,1,2,4],
        [2,0,5,5,1,6,3],
        [5,4,1,2,0,6,6],
        [0,1,2,3,4,5,1],
    ]
    agent = Agent(grid=grid)
    for solution in agent.dfs():
        print(f"Solution: ",*solution, sep="\n")
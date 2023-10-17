import numpy as np
class Environment():
    def __init__(self):
        #11x14 grid with 1 cell thick border
        self.grid = np.zeros((13,16))
        #set border cells to 1
        self.grid[0,:] = 1
        self.grid[12,:] = 1
        self.grid[:,0] = 1
        self.grid[:,12:] = 1
        self.grid[11,6:9] = 1
        self.grid[10,6:9] = 1
        self.grid[5:8,12:15] = 0
        #add obstacle
        self.grid[5:8,3:9] = 1
        self.grid[6:8,5:7] = 0
        
        #place robot
        self.robot = (2,6)

    #def percieve(self):
    #    #return the 3x3 grid around the robot
    #    x,y = self.robot
    #    contour = self.grid[x-1:x+2,y-1:y+2].flatten()
    #    # remove the center cell
    #    s = np.delete(contour,4)
    #    return s

    #def percieve(self):
    #    x,y = self.robot
    #    return [ self.grid[x+i,y+j] for i in range(-1,2) for j in range(-1,2) if not (i == 0 and j == 0) ]

    def percieve(self):
        x,y = self.robot
        s = []
        for i in range(-1,2):
            for j in range(-1,2):
                if (i == 0 and j == 0):
                    continue
                s.append(self.grid[x+i,y+j])
        return s

    def move(self,action):
        new_pos = list(self.robot)
        if   action == "Up":
            new_pos[0] -= 1
        elif action == "Down":
            new_pos[0] += 1
        elif action == "Left":
            new_pos[1] -= 1
        elif action == "Right":
            new_pos[1] += 1
        new_pos = tuple(new_pos)
        if self.grid[new_pos] != 1:
            self.robot = new_pos

    def __str__(self):
        s = ""
        for i,row in enumerate(self.grid):
            for j,cell in enumerate(row):
                s += "R|" if (i,j) == self.robot else "#|" if cell == 1 else " |"
            s += "\n"
        return s

class Robot():
    def __init__(self, environment):
        self.environment = environment

    def action(self):
        perception = self.environment.percieve()
        if sum(perception) == 0:
            self.environment.move("Up")

        # 0 1 2 3 4 5 6 7
        # 0 1 2 4 7 6 5 3
        perception = [perception[i] for i in [0,1,2,4,7,6,5,3]]

        x1 = sum(perception[1:3] ) > 0
        x2 = sum(perception[3:5] ) > 0
        x3 = sum(perception[5:7] ) > 0
        x4 = perception[0]+ perception[7] > 0

        if x1 and not x2:
            self.environment.move("Right")
        elif x2 and not x3:
            self.environment.move("Down")
        elif x3 and not x4:
            self.environment.move("Left")
        elif x4 and not x1:
            self.environment.move("Up")

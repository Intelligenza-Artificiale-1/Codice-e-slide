import random
#Vacuum world environment
class Vacuum():
    def __init__(self, environment):
        self.environment = environment
        self.location = "A"

    def perceive(self):
        return self.environment.perceive(self)

    def get_location(self):
        return self.location

    def move(self, direction):
        if direction == 'Right' and self.location == 'A':
            self.location = 'B'
        elif direction == 'Left' and self.location == 'B':
            self.location = 'A'
        else:
            pass

    def clean(self):
        self.environment.clean(self)


class Environment():
    def __init__(self):
        self.locationCondition = {'A':'Dirty', 'B':'Dirty'} 

    def perceive(self, agent):
        return self.locationCondition[agent.location]

    def clean(self, agent):
        self.locationCondition[agent.location] = 'Clean'

    def __str__(self):
        return str(self.locationCondition)


class TableVacuum(Vacuum):
    def __init__(self, environment):
        super().__init__(environment)

    def agentFunction(self):
        status = self.perceive()
        if  self.get_location() == 'A' and status == 'Clean':
            self.move('Right')
        elif self.get_location() == 'A' and status == 'Dirty':
            self.clean()
        elif self.get_location() == 'B' and status == 'Clean':
            self.move('Left')
        elif self.get_location() == 'B' and status == 'Dirty':
            self.clean()

class ReflexVacuum(Vacuum):
    def __init__(self, environment):
        super().__init__(environment)

    def agentFunction(self):
        status = self.perceive()
        if status == 'Dirty':
            self.clean()
        else:
            self.move( 'Right' if self.get_location() == 'A' else 'Left')

class BlindVacuum(Vacuum):
    def __init__(self, environment):
        super().__init__(environment)

    def get_location(self):
        return ""

    def agentFunction(self):
        status = self.perceive()
        if status == 'Dirty':
            self.clean()
        else:
            self.move( 'Right' if random.randint(0,1) == 0 else 'Left')

class ModelVacuum(Vacuum):
    def __init__(self, environment):
        super().__init__(environment)
        self.model = {'A':None, 'B':None}

    def agentFunction(self):
        status = self.perceive()
        self.model[self.get_location()] = status
        if self.model['A'] == self.model['B'] == 'Clean':
            print('Done')
        elif status == 'Dirty':
            self.clean()
        else:
            self.move( 'Right' if self.get_location() == 'A' else 'Left')

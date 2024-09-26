import random
#Vacuum world environment

class Environment():
    def __init__(self):
        pass

    def _step(self):
        pass

    @property
    def status(self) -> str:
        pass

    @property
    def vacuum_location(self) -> str:
        pass

    @vacuum_location.setter
    def vacuum_location(self, location):
        pass

    def move_vacuum(self, direction: str):
        """
        Moves the vacuum to the specified direction if the move is valid.
        Parameters:
        direction (str): The direction to move the vacuum. Valid values are 'R' and 'L'.
        """
        pass

    def clean(self):
        pass

    def __str__(self) -> str:
        pass


# Vacuum agents
class Vacuum():
    def __init__(self, environment):
        pass

    def perceive(self) -> str:
        pass

    def get_location(self) -> str:
        pass

    def apply_action(self, action : str):
        if action == 'R' or action == 'L':
            self.environment.move_vacuum(action)
        elif action == 'C':
            self.environment.clean()
        else:
            print('Invalid action')

    def agent_function(self):
        pass

class ManualVacuum(Vacuum):
    def __init__(self, environment):
        super().__init__(environment)

    def agent_function(self):
        pass

class TableVacuum(Vacuum):
    def __init__(self, environment):
        super().__init__(environment)
        self.table = {
            (('A','Clean'),):'R',
            (('A','Dirty'),):'C',
            (('B','Clean'),):'L',
            (('B','Dirty'),):'C',
            (('A','Dirty'),('A','Clean')):'R',
            (('A','Clean'),('B','Dirty')):'C',
            (('B','Clean'),('A','Dirty')):'C',
            (('B','Dirty'),('B','Clean')):'L',
            (('A','Dirty'),('A','Clean'),('B','Dirty')):'C',
            (('B','Dirty'),('B','Clean'),('A','Dirty')):'C'
        }
        self.percept_sequence = []

    def agent_function(self):
        while True:
            pass
        

class ReflexVacuum(Vacuum):
    def __init__(self, environment):
        super().__init__(environment)

    def agent_function(self):
        pass


class BlindVacuum(Vacuum):
    def __init__(self, environment):
        super().__init__(environment)

    def perceive(self) -> str:
        return ""
    
    def get_location(self) -> str:
        return ""

    def agent_function(self):
        pass


class ModelVacuum(Vacuum):
    def __init__(self, environment):
        super().__init__(environment)
        self.model = {'A':None, 'B':None}

    def agent_function(self):
        while self.model != {'A':'Clean', 'B':'Clean'}:
            pass



# Altri tipi di ambienti

class NonDeterministicEnvironment(Environment):
    def __init__(self):
        super().__init__()

    @Environment.vacuum_location.setter
    def vacuum_location(self, location):
        pass

    def clean(self):
        pass

class DynamicEnvironment(Environment):
    def __init__(self):
        super().__init__()

    def step(self):
        super().step()

class NoisyEnvironment(Environment):
    def __init__(self):
        super().__init__()
    
    @property
    def status(self):
        return self._location_condition[self._vacuum_location]

    @property
    def vacuum_location(self):
        return self._vacuum_location
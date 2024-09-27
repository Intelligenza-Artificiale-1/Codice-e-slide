import random
#Vacuum world environment

class Environment():
    def __init__(self):
        self._world = {'A':'Dirty', 'B':'Dirty'} 
        self._vacuum_location = 'A'

    def _step(self):
        print(self)

    @property
    def status(self) -> str:
        return self._world[self._vacuum_location]

    @property
    def vacuum_location(self) -> str:
        return self._vacuum_location

    @vacuum_location.setter
    def vacuum_location(self, location : str):
        self._vacuum_location = location

    def move_vacuum(self, direction: str):
        """
        Moves the vacuum to the specified direction if the move is valid.
        Parameters:
        direction (str): The direction to move the vacuum. Valid values are 'R' and 'L'.
        """
        if direction == 'R':
            self.vacuum_location = 'B'
        elif direction == 'L':
            self.vacuum_location = 'A'
        else:
            assert False, "Invalid direction"
        self._step()

    def clean(self):
        self._world[self._vacuum_location] = 'Clean'
        self._step()

    def __str__(self) -> str:
        return f"Vacuum in location {self._vacuum_location}, state: {self._world}"


# Vacuum agents
class Vacuum():
    def __init__(self, environment : Environment):
        self.environment = environment

    def perceive(self) -> str:
        return self.environment.status

    def get_location(self) -> str:
        return self.environment.vacuum_location

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
    def __init__(self, environment : Environment):
        super().__init__(environment)

    def agent_function(self):
        print("Possible actions: 'R' (Right), 'L' (Left), 'C' (Clean), 'P' (Perceive), 'Q' (Quit)")
        while (x := input('\tEnter action: ').upper()) != 'Q':
            if x == 'P':
                print(f"\t\tLocation: {self.get_location()}, Status: {self.perceive()}")
            else:
                self.apply_action(x)

class TableVacuum(Vacuum):
    def __init__(self, environment : Environment):
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
            new_perception = (self.get_location(), self.perceive())
            self.perception_sequence.append(new_perception)
            if (key:=tuple(self.perception_sequence)) not in self.table:
                break
            action = self.table[key]
            self.apply_action(action)
        

class ReflexVacuum(Vacuum):
    def __init__(self, environment : Environment):
        super().__init__(environment)

    def agent_function(self):
        while input('Continue? (Y/n): ').lower() != 'n':
            if self.perceive() == 'Dirty':
                self.apply_action('C')
            elif self.get_location() == 'A':
                self.apply_action('R') 
            else:
                self.apply_action('L')


class BlindVacuum(Vacuum):
    def __init__(self, environment : Environment):
        super().__init__(environment)

    def perceive(self) -> str:
        return ""
    
    def get_location(self) -> str:
        return ""

    def agent_function(self):
        while input('Continue? (Y/n): ').lower() != 'n':
            self.apply_action('C')
            self.apply_action( 'R' if random.randint(0,1) == 0 else 'L')


class ModelVacuum(Vacuum):
    def __init__(self, environment : Environment):
        super().__init__(environment)
        self.model = {'A':None, 'B':None}

    def agent_function(self):
        while self.model != {'A':'Clean', 'B':'Clean'}:
            pos, state = self.get_location(), self.perceive()
            self.model[pos] = state
            if state == 'Dirty':
                self.apply_action('C')
            elif pos == 'A':
                self.apply_action('R')
            else:
                self.apply_action('L')



# Altri tipi di ambienti

class NonDeterministicEnvironment(Environment):
    def __init__(self):
        super().__init__()

    @Environment.vacuum_location.setter
    def vacuum_location(self, location : str):
        if random.random() > 0.9:
            self.vacuum_location = location

    def clean(self):
        if random.random() > 0.9:
            super().clean()

class DynamicEnvironment(Environment):
    def __init__(self):
        super().__init__()

    def step(self):
        if random.random() > 0.6:
            if random.random() > 0.5:
                self._world['A'] = 'Dirty'
            else:
                self._world['B'] = 'Dirty'
        super().step()

class NoisyEnvironment(Environment):
    def __init__(self):
        super().__init__()
    
    @property
    def status(self):
        if random.random() > 0.3:
            return self._world[self._vacuum_location]
        else:
            return 'Clean'

    @property
    def vacuum_location(self):
        if random.random() > 0.2:
            return self._vacuum_location
        else:
            return 'A' if random.random() > 0.5 else 'B'
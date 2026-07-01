class car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    def move(self):
        print("drive")
class boat:    
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    def move(self):
        print("sail")

class plane:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    def move(self):
        print("Fly!")
    
Car1 = car("bmw","mustang")
Boat1 = boat("ss","dd")
Plane1 = plane("kingfisher","3323")


for x in (Car1, Boat1, Plane1 ):
    x.move()     
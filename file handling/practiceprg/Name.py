class Name:
    def __init__(self,name,college_name):
        self.name = name
        self.college_name = college_name
        
s = Name("darshan","ubdt")
print(s.name)
print(s.college_name)

for i in range(len(s.name) - 1, -1, -1):
    print(s.name[i], end=" ")


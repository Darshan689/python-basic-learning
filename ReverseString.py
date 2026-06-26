string = input("Enter a string: ")

reversed_string = ""

# Traverse the string from end to beginning
for i in range(len(string) - 1, -1, -1):
    reversed_string += string[i]

print("Reversed String:", reversed_string)
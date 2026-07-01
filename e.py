file = open("employee.txt", "w")

name = input("Enter employee name: ")
department = input("Enter department: ")
salary = input("Enter salary: ")

file.write(f"Name: {name} \n")
file.write(f"Department: {department}\n")
file.write(f"Salary: {salary} \n")

file.close()

print("Employee details saved.")
# name = "darshan"
# age =  23
#price = 23.33
# print(type(name))
# print(type(age))
#print(type(price))
#b = 20
#print(a==b)
#print(a!=b)
#print(a>b)
# # # # # print(a<b)
# # # # # print(a>=b)
# # # # # print(a<=b)

# # # # num = 20
# # # # num +=10
# # # # print(num)

# # # num  = 30 
# # # num **=56
# # # print(num)

# # # print(2**10)

# # #type casting
# # a = float("2")
# # b= 4.34
# # print(a + b)
# s = int(input("enter the first number:"))
# p = int(input("enter the second number:"))
# print("sum of two number is:",s+p)
# from collections import deque
# queue = deque(["darshan","ss","tt"])
# queue.append("pp")
# queue.append("xx")

# person = queue.leftpop()
# pront(person)

#string:
# Str1 = "Good moring sir.\n my name is darshan naik hs from shivamoga. \n  i have completed my completed  b tech in ubdt collge of enginnering with cgpa of 8.0  "
# print(Str1)

# s1 = "darshan"
# s2 = "ss"
# s3 ="xx"
# s4 = str(11)
# print(s1+s2+s3+s4)
# print(len(s1+s2+s3+s4))

# s = "darshan"
# print(s[0:])
# print(s[-2:-1])

# x = "My name is darshan i curently study in BE engineering ubdt collge of engineering in davangere"
# print(x.endswith("BE"))

# time = 12
# if (time == 11):
#     print("its time for brek:")
# if (time == 10):
#     print("its time for  drinks braks:")
# if (time == 12):
#     print("its time for drinking some water:")
# else:
#     print("its your time to leave:")
# marks = int(input("enter your marks:"))   
# if (marks >= 90):
#     print("you got 'A'grade:")
# elif (marks >=80):
#     print("you got 'B'grade:")
# elif (marks >=70):
#     print("you got 'C' grade:")
# elif (marks>60):
#     print("you got 'D' grade:")
# else:
#     print("your are loosers")


# num = int(input("enter your number:"))
# rem= num % 2
# if (rem == 0):
#     print("even")
# else:
#     print("odd")
    
# file = open("q.txt","w")
# name = input("enter your name:")
# department = input("enter your department:")
# salary = input("enter your salary:")
# file.write(f"Name:{name}\n")
# file.write(f"Department:{department}\n")
# file.write(f"Salary:{salary}\n")
# file.close()


# Student = {
#     "name":"darshan hs",
#     "subject" : {
#         "maths":29,
#         "chemistry":56,
#         "biology":45,
#         "physics":45
        
#     }
# }
# print(len(Student.keys()))

x = {
    "name": "darshan",
    "age":21,
    "weight":50,
    "height":150,
    "sex":"male",
    "biodata":{
        "qualification":"Btech",
        "skillset": "python,sql",
        
    }
     
 }
x.update({"village":"goa"})
print(x)
print(x.__subclasshook__)

c = {}
print(type(c))

x =  ()
print(type(x))
v = set({})
print(type(v))

#Union in sets 
set1 = {1,2,3}
set2 = {2,3,4}
print(set1.union(set2))

# intersection in sets
set1 = {1,2,3}
set2 = {2,3,4}
print(set1.intersection(set2))

    
    
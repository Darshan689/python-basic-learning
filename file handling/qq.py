# file = open("q.txt","r")
# s = file.read()
# print(s)
# file.close()

# Student = ["darshan","sudeep","surya",1,2,3]
# print(Student)
# Student[0] = "ss"
# print(Student)

# #slicing in list values:

# x = [23,34,45,46,78,89]
# print(x[1:4])

# fruits = ["mango"]
# print(fruits[1:4])

# #sorting  the elements oin list:
# z = [5,3,4,2,6]
# print(z.sort())
# print(z)

# z = [4,3,2]
# z.sort(reverse = True)
# print(z)

# p =["darshan","shivu", "bharath"]
# p.insert(2,"surya")
# print(p)
# copy the  elements in list:

# q = [1,2,3,4,5,6]
# print(q)
# q.copy()
# print(q)


# ### tuples###
# tuple = (1,2,3,4,2)
# print(type(tuple))


# movies =[]
# mov1  = input("enter your 1st movies:")
# mov2  = input("enter your2nd movies:")
# mov3  = input("enter your 3rd movies:")
# movies.append(mov1)
# movies.append(mov2)
# movies.append(mov3)
# print(movies)


# list1 = [1,2,1]
# list2 = [1,2,3]

# copy_list1 = list1.copy()
# copy_list1.reverse()

# if (copy_list1 == list1):
#     print("its a palindrome:")
# else:
#     print("its not  palindrome:")
    
    
# copy_list2 = list2.copy()
# copy_list2.reverse()

# if (copy_list2 == list2):
#     print("its a palindrome:")
# else:
 
#     print("its not palindrome:")
    
    
    
# a = 10
# b = 20
# print("a =",a, "b=",b)
# a,b = b,a
# print("a",a,"b=",b)

# Balance = 200000
# withdraw_amount =299999

# if (withdraw_amount<=Balance):
#     print("amount withdraw sucesssfully")

# else:
#     print("insuffiecnt fund or amount")


# arrival = [900, 940, 950, 1100, 1500, 1800]
# departure = [910, 1200, 1120, 1130, 1900, 2000]

# arrival.sort()
# departure.sort()
# i = 0
# j = 0
# platform_needed = 0
# maximum_platform = 0
# while i< len(arrival) and j< len(departure):
#     if arrival [i]<= departure[j]:
#         platform_needed +=1
#         i+=1
#         maximum_platform = max(maximum_platform,platform_needed)
#     else:
#         platform_needed -=1
#         j+=1
# print(maximum_platform)

# dict = {
#     "name": "darshan",
#     "age":21,
#     "subject":"python basic",
#     "cgpa":8.0
# }

# dict["name"]= "ss" # overwrite in main dictionary 
# print(dict)

# L = [1,2,3,4,5,6,7]
# L.append(1000)
# L.insert(2,199)
# L.sort()

# print(L)
s = "Darshan"+"ss"
print(s)

s = 12
t = int(13.345)
sum = s+t
print(sum)

Q = [1,2,3,4,5,5,6]
print(Q[1:4])
print(Q[6])
print(Q[::-1])
print(Q[0:0])
print(Q[6:6])

tuple = ("m","a","m")
copy_tuple= tuple.copy()
copy_tuple.reverse()
if (copy_tuple == copy_tuple):
    print("its a palindrome:")
else:
    print("its not a palindrome")
list1 = [1,2,1]
list2 = [1,2,3]
copy_list1 = list1.copy()
copy_list1.reverse()

if (copy_list1 == list2):
    print("its a palindrome:")
    
else:
    print("its not a palindrome")
    
    
    
n = 50
sum = 0
for i in range(1,n+1):
    sum +=i
print("total sum is:", sum)

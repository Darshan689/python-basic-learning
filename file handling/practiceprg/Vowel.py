text =  input("enter  a  string:")
Vowels = ("aeiouAEIOU")
count = 0
for i in  text:
    if i in Vowels:
        count = count + 1
print("count  the vowels:",count)
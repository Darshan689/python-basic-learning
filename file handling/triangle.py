l = [1,2,3,4,5,6,7,8]
x= 8
pos = -1
for s in range(0,(len(l))):
    if l[s] == x:
        pos = s
print(pos)

x = ['siri','dashan','chethu','nandini','pratap']
p = 'chethu'
pos =-1
for i in range(len(x)):
    if x[i] == p:
        pos = i
        
print(pos)


arr = [10,20,30,40,50]
ele = 50
pos =-1
low =0
high = len(arr)-1
while  low<=high:
    mid = (low-high)//2
    
    if arr[mid]== ele:
        pos = mid
        break

    elif arr[mid] > ele:
        high = mid - 1
    else:
        low = mid + 1
print(pos)   



class BankAccount:
    self.balance = 10000
    
    def _show_balance(self):
        print(f"balance:{self.balance}")    
    
    
    def _update_balance(self,amount):
        self.balance += amount
        
        
    def deposit(self,amount):
        if amount > 0:
            self.__update_balance(amount)
            self._show_balance()
        else:
            print("Invalid deposite amount:")
amount = BankAccount()
amount._show_balance()

amount.deposit(500)
            
            
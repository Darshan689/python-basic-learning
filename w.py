class Manager:
    def work(self):
        print("manager momitoring the all employee")
        
class Developer:
    def work(self):
        print("develpoer developing the projects")

class Tester:
    def work(self):
        print("tester test in projects:")

s1 =   [Manager(), Tester(), Developer()]
for s1 in s1:
    s1.work()
    
    
class Email:
    def send(self):
        print("sending emails")

class message:
    def send(self):
        print("sending whatsapp messsages:")

class whatsapp:
    def send(self):
        print("sending whatsapp message ")
        
notifications = [message(),Email(),whatsapp()]
for notification  in notifications:
    notification.send()
          

        
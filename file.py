import threading
import time
#function to simulate downloading a file
def download(file_name):
    print(f"Starting download :{file_name}")
    time.sleep(3)
    print(f"finished dowloading: {file_name}")
    
#create threads

t1 = threading.Thread(target=download,args=("file1.pdf",))
t2 = threading.Thread(target=download,args=("file2.pdf",))
t3 = threading.Thread(target=download,args=("file3.pdf",))

#start all thread

t1.start()
t2.start()
t3.start()

#wait for all dowlaods to finish
t1.join()
t2.join()
t3.join()


print("all downloads compelted")
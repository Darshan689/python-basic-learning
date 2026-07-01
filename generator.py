import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Decorator
def logger(func):
    def wrapper():
        logging.info(f"{func.__name__} started")
        func()
        logging.info(f"{func.__name__} finished")
    return wrapper

# Function
@logger
def greet():
    print("Hello, Darshan!")

# Calling the function
greet()
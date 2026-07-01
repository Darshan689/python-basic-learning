a = int(input ("a:"))
b = int(input( "b :"))

try:
    s = (a/b)
except Exception as e:
    print(f"Error: {e}")
    b = int(input("b:"))
    print(a/b)
finally:
    print("error occoured wheather the proramm excuted or not!:")
    
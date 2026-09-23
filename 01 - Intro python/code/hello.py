def valori():
    for i in range(1):
        print("Sto calcolando roba")
        yield i
        print("Ho calcolato roba")
        return -1

v = valori()
print(v)

while( x := input("Vuoi altri valori? Y/n")) != "n":
    print(next(v,None))

v = valori()
print(next(v))

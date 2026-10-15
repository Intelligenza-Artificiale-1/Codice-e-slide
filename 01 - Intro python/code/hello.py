def funzione_return():
    a = 1
    b = 20
    for _ in range(b):
        a=a+1
        return a
        print(f"{id(a)=}, {a=} \t {id(b)=}, {b=}")

def funzione_yield():
    a = 1
    b = 20
    for _ in range(b):
        a=a+1
        yield a
        print(f"{a=} \t {b=}")


d = {"Pierfranco":35000, "Gigio":100, "Annamaria": 40000}

print(any(d[key] < 10000 for key in d))
print(any(val < 10000 for val in d.values()))



a = [1, 2, 3, 4, 5]
for i, num in enumerate(a):
    if (num%2)!=0:
        num=0
        a[i]=num
    print(num)

print(a)



l1 = [1, 2, 3, 4, 5]
l2 = [3,10, 4, 4, 4]

flag = True
min_len=min(len(l1), len(l2))
for i in range(min_len):
    if l1[i] > l2[i]:
        flag = False

print(flag)

flag = True
for x1, x2 in zip(l1,l2):
    if x1 > x2:
        flag = False

print(flag)


print(all( x1 >= x2 for x1, x2 in zip(l1, l2)))

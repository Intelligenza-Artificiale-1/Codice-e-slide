def is_sorted1(l):
    return l==sorted(l)

def is_sorted4(l):
    return not any( x>y for x,y in zip(l,l[1:]))

def is_sorted2(l):
    for i in range(len(l)-1):
        if l[i] > l[i+1]:
            return False
    return True

def is_sorted3(l):
    return all(l[i] < l[i+1] for i in range(len(l)-1))

def fib():
    a,b = 0,1
    while b<5:
        yield b
        a, b = b, a+b


def main():
    print(all(x % 2 == 0 for x in fib()))

if __name__ == '__main__':
    main()
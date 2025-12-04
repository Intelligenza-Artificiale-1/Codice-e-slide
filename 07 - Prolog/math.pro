sum(X,0,X).
sum(X,s(Y),Z) :- sum(s(X),Y,Z).  % x+(y+1) = (x+1) + y

diff(X,Y,Z) :- sum(Y,Z,X).       % x-y=z -> x-y-z = 0 -> y+z=x

mul(_,0,0).
mul(X,s(Y),Z) :-                 % x*(y+1) = x*y + x
    mul(X,Y,Z1),
    sum(X,Z1,Z).

div(X,Y,Z):-                     % X/Y=Z -> X=Y*Z
    mul(Y,Z,X).

abs(X,X):- X>=0.
abs(X,Y):- X<0, Y is -X.


mcd(X,0,X).
mcd(X,Y,Z):-
    Y>0,
    Q is X mod Y,
    mcd(Y, Q, Z).


print_n(_,0).
print_n(S,N):-
    N>0,
    writeln(S),
    N1 is N-1,
    print_n(S, N1).


for(Start,Stop,Inc, F):-
    Start < Stop,
    F,
    Start1 is Start + Inc,
    for(Start1, Stop, Inc, F).


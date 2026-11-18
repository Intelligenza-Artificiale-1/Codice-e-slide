% Archi:    A-B, A-F, A-G, B-C, C-D, C-E, C-F, F-G
colore(rosso).
colore(verde).
colore(blu).

coppia_valida(X, Y) :- 
    colore(X),
    colore(Y), 
    X \= Y.

colorazione(A, B, C, D, E, F, G) :-
    coppia_valida(A, B), coppia_valida(A, F),
    coppia_valida(A, G), coppia_valida(B, C),
    coppia_valida(C, D), coppia_valida(C, E),
    coppia_valida(C, F), coppia_valida(F, G).

regioni([a,b,c,d,e,f,g]).

archi([a-b, a-f, a-g, b-c,
       c-d, c-e, c-f, f-g]).

colore(rosso).
colore(verde).
colore(blu).


% Assegnamento è una lista [Regione-Colore, ...]
colorazione(Assegnamento) :-
    regioni(Regioni),
    assegna(Regioni, Assegnamento),
    archi(Archi),
    archi_ok(Archi, Assegnamento).

assegna([], []).
assegna([R|Rs], [R-C|As]) :-
    colore(C),
    assegna(Rs, As).

% archi_ok(+Archi, +Assegnamento)
archi_ok([], _).
archi_ok([A-B|Resto], Assegnamento) :-
    colore_di(A, Assegnamento, CA),
    colore_di(B, Assegnamento, CB),
    CA \= CB,
    archi_ok(Resto, Assegnamento).

% colore_di(+Regione, +Assegnamento, ?Colore)
colore_di(R, [R-C|_], C).
colore_di(R, [_|Resto], C) :-
    colore_di(R, Resto, C).
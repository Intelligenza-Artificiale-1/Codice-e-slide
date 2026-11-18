% Fatti: quali colori sono disponibili? 
colore(rosso).
colore(verde).
colore(blu).

% Regola: due colori scelti dal dominio sono diversi 
coppia_valida(X, Y) :-
    colore(X),
    colore(Y),
    X \= Y.
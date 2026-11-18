:-include("02.pro").

colorazione_estesa(A,B,C,D,E,F,G,H):-
    colorazione(A, B, C, D, E, F, G),
    coppia_valida(H,G),
    coppia_valida(H,E).
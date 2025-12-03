genitore(a,b).
genitore(b,c).
antenato(X,Y) :- genitore(X,Y).
antenato(X,Y) :- genitore(X,Z), antenato(Z,Y).
%antenato(X,Y) :- antenato(Z,Y), genitore(X,Z). RICORSIONE INFINITA
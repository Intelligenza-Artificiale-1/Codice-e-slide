#Braccio robotico  che tenta di sollevare un blocco:
#    M = c'è movimento
#    B = la batteria è carica
#    S = il blocco è sollevabile
#    I = l'indicatore è acceso

import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD

# Define the structure.
edges = [('S','M'), ('B','M'), ('B','I')]

cpt_i = TabularCPD(variable='I', variable_card=2, 
                   values=[[0.9, 0.05],
                           [0.1, 0.95]],
                   evidence=['B'], evidence_card=[2])

cpt_m = TabularCPD(variable='M', variable_card=2,
                   values=[[1, 1, 0.95, 0.1],
                           [0, 0, 0.05, 0.9]],
                   evidence=['B', 'S'], evidence_card=[2, 2])

cpt_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.05],[0.95]])
cpt_b = TabularCPD(variable='B', variable_card=2,
                   values=[[0.1],[0.9]])

DAG = bn.make_DAG(edges, CPD=[cpt_i, cpt_m, cpt_s, cpt_b])
#bn.plot(DAG)
#bn.print_CPD(DAG)

bn.inference.fit(DAG, variables=['M'], evidence={'S':1, 'I':0, 'B':1})

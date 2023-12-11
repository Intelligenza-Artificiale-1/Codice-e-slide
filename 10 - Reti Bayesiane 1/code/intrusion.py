# Abbiamo installato un nuovo antifurto, che è abbastanza affidabile ma occasionalmente scatta anche per piccoli terremoti
# Abbiamo due vicini, John e Mary, che hanno promesso di telefonare se scatta l’allarme
# John chiama quasi sempre quando scatta l’allarme, ma a volte scambia lo squillo del telefono per l’allarme
# Mary non sempre sente l’allarme
#
#	Variabili: \codeinline{['Intrusione', 'Terremoto', 'Allarme', 'John', 'Mary']}
#
#	Inferenza: \codeinline{bn.inference.fit(DAG, variables=['Intrusione'], evidence=\{'John':1, 'Mary':0\})}

import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD

# Define the structure.
edges = [('Intrusione', 'Allarme'), ('Terremoto', 'Allarme'), ('Allarme', 'John'), ('Allarme', 'Mary')]

cpt_intrusione = TabularCPD(variable='Intrusione', variable_card=2,
                            values=[[0.999], [0.001]])
cpt_terremoto  = TabularCPD(variable='Terremoto', variable_card=2,
                            values=[[0.998], [0.002]])

cpt_allarme    = TabularCPD(variable='Allarme', variable_card=2,
                            values=[[0.999, 0.71, 0.06, 0.05],
                                    [0.001, 0.29, 0.94, 0.95]],
                            evidence=['Intrusione', 'Terremoto'],
                            evidence_card=[2, 2])

cpt_john       = TabularCPD(variable='John', variable_card=2,
                            values=[[0.95, 0.1],
                                    [0.05, 0.9]],
                            evidence=['Allarme'], evidence_card=[2])

cpt_mary       = TabularCPD(variable='Mary', variable_card=2,
                            values=[[0.99, 0.3],
                                    [0.01, 0.7]],
                            evidence=['Allarme'], evidence_card=[2])

dag = bn.make_DAG(edges, CPD=[cpt_intrusione, cpt_terremoto, cpt_allarme, cpt_john, cpt_mary])
#bn.plot(dag)
bn.inference.fit(dag, variables=['Intrusione'], evidence={'John':1, 'Mary':0})

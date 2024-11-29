import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD

edges = [('Giocare', 'Pioggia'), ('Giocare', 'Vento')]
DAG = bn.make_DAG(edges)
#bn.plot(DAG)

cpt_giocare = TabularCPD(variable='Giocare', variable_card=2, values=[[0.1], [0.9]])
cpt_pioggia = TabularCPD(variable='Pioggia', variable_card=2, evidence=['Giocare'], evidence_card=[2], values=[[0.3, 0.6], [0.7, 0.4]])
cpt_vento = TabularCPD(variable='Vento', variable_card=2, evidence=['Giocare'], evidence_card=[2], values=[[0.85 ,0.95], [0.15, 0.05]])

DAG = bn.make_DAG(edges, CPD=[cpt_giocare, cpt_pioggia, cpt_vento])

result = bn.inference.fit(DAG, variables=['Pioggia'], evidence={'Vento':0})
result = bn.inference.fit(DAG, variables=['Pioggia'], evidence={'Vento':1})
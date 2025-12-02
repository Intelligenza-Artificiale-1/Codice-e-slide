import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD

edges = [('Disp','Acquisto'),('Pub','Prec'),('Sugg','Prec'),('Prec','Acquisto')]

# Assumo che 1=telefono, 0=pc
disp_cpt = TabularCPD(variable='Disp', variable_card=2, values=[[0.65],[0.35]])
pub_cpt = TabularCPD(variable='Pub', variable_card=2, values=[[0.97],[0.03]])
sugg_cpt = TabularCPD(variable='Sugg', variable_card=2, values=[[0.30],[0.70]])

prec_cpt = TabularCPD(variable='Prec', variable_card=2, evidence=['Pub','Sugg'], evidence_card=[2,2],
                      values=[  [0.9, 0.9,0.9, 0.6],
                                [0.1, 0.1, 0.1, 0.4]])
acq_cpt = TabularCPD(variable='Acquisto', variable_card=2, evidence=['Disp','Prec'], evidence_card=[2,2],
                      values=[  [0.8, 0.5, 0.9, 0.9],
                                [0.2, 0.5, 0.1, 0.1]])

dag = bn.make_DAG(edges, CPD=[disp_cpt, pub_cpt , sugg_cpt, prec_cpt, acq_cpt])
bn.inference.fit(model=dag, variables=['Prec','Acquisto'], evidence={'Sugg':1})
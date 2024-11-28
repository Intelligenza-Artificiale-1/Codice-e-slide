import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD

# Definizione della rete bayesiana
edges = [('mobile','acquisto'),('precedente','acquisto'),('pubblicita','precedente'),('suggestionabile','precedente')]


cpd_mobile = TabularCPD('mobile', 2, [[0.65], [0.35]])
cpd_suggestionabile = TabularCPD('suggestionabile', 2, [[0.3], [0.7]])
cpd_pubblicita = TabularCPD('pubblicita', 2, [[0.97], [0.03]])

cpd_precedente = TabularCPD('precedente', 2, [[0.9, 0.9, 0.9, 0.6],
                                                [0.1, 0.1, 0.1, 0.4]],
                                              evidence=['pubblicita','suggestionabile'], evidence_card=[2,2])

cpd_acquisto = TabularCPD('acquisto', 2, [[0.8, 0.5, 0.9, 0.9],
                                            [0.2, 0.5, 0.1, 0.1]],
                                            evidence=['mobile','precedente'], evidence_card=[2,2])

model = bn.make_DAG(edges, CPD=[cpd_mobile, cpd_suggestionabile, cpd_pubblicita, cpd_precedente, cpd_acquisto])
#bn.plot(model)

bn.inference.fit(model, variables=['acquisto'], evidence={'suggestionabile': 1, 'mobile': 1})
bn.inference.fit(model, variables=['acquisto'], evidence={})
bn.inference.fit(model, variables=['pubblicita'], evidence={'acquisto': 0})
bn.inference.fit(model, variables=['acquisto'], evidence={'pubblicita': 1})
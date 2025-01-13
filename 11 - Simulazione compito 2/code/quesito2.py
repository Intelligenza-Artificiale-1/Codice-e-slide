import random
import pandas as pd
import os, sys
import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD


edges = [('West Wind', 'Red Sky'), ('Low Pressure', 'Barometer'), ('Low Pressure', 'Rain'), ('West Wind', 'Rain')]


#West Wind = 1 -> Vento da ovest verso est
cpd_west_wind = TabularCPD(variable='West Wind', variable_card=2, values=[[0.9],[ 0.1]])
cpd_low_pressure = TabularCPD(variable='Low Pressure', variable_card=2, values=[[0.7],[0.3]])

# Barometer = 1 -> Segna alta pressione
p_barometer_low_given_low_pressure = 0.7*0.4/0.3
p_barometer_high_given_low_pressure = 1 - p_barometer_low_given_low_pressure
cpd_barometer = TabularCPD(variable='Barometer', variable_card=2, evidence=['Low Pressure'], evidence_card=[2], values=[[0.1, p_barometer_low_given_low_pressure], [0.9, p_barometer_high_given_low_pressure]])
cpd_red_sky = TabularCPD(variable='Red Sky', variable_card=2, evidence=['West Wind'], evidence_card=[2], values=[[0.6, 0.2], [0.4, 0.8]])

cpd_rain = TabularCPD(variable='Rain', variable_card=2, evidence=['West Wind', 'Low Pressure'], evidence_card=[2, 2], values=[[0.8, 0.9, 0.3, 0.7], [0.2, 0.1, 0.7, 0.3]])

dag  = bn.make_DAG(edges, CPD=[cpd_west_wind, cpd_barometer, cpd_rain, cpd_low_pressure, cpd_red_sky])
print(bn.inference.fit(dag, variables=['Rain'], evidence={'Red Sky': 1, 'Low Pressure': 0}))

import random
import pandas as pd
import os, sys
import bnlearn as bn

def create_tennis_data(rows=100):
    #get current dir
    dir_path = os.path.dirname(os.path.abspath(__file__))
    #write to file
    with open(dir_path+"/tennis.csv", "w") as f:
        f.write("rain,humid,wind,play\n")
        for _ in range(rows):
            rain  = random.choice([0, 0, 0, 1])
            humid = random.choice([0, 1, 1])
            wind  = random.choice([0, 1])

            play = 0
            if rain == 1:
                play = 1 if random.random() < 0.01 else 0
            elif humid == 1 and wind == 0:
                play = 1 if random.random() < 0.35 else 0
            elif humid == 0 and wind == 0:
                play = 1 if random.random() < 0.9 else 0
            elif humid == 1 and wind == 1:
                play = 1 if random.random() < 0.5 else 0
            else:
                play = 1 if random.random() < 0.7 else 0
            f.write(f"{rain},{humid},{wind},{play}\n")

def read_data():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(dir_path+"/tennis.csv")
    return data

def create_dag():
    edges = [('play', 'rain'), ('play', 'humid'), ('play', 'wind')]
    DAG = bn.make_DAG(edges)
    return DAG

def learn_cpd(dag, df):
    dag = bn.parameter_learning.fit(dag, df, methodtype='maximumlikelihood')
    bn.print_CPD(dag)
    return dag
    

def predict(DAG, variables, evidence):
    # get the conditional probability of the variables given the evidence
    cpd = bn.inference.fit(DAG, variables=variables, evidence=evidence)
    return cpd



if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(dir_path+"/tennis.csv"):
        create_tennis_data(rows=50)
    data = read_data()

    DAG = create_dag()
    DAG = learn_cpd(DAG, data)
    predict(DAG, variables=['play'], evidence={ 'rain':0, 'wind': 1})


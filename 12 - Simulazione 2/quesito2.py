import random
import pandas as pd
import os, sys
import bnlearn as bn

#set seed
random.seed(42)

def get_weather_data(rows=200):
    #get current dir
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, "weather.csv")
    #if file does not exist, create it
    if not os.path.exists(file_path):
        #write to file
        with open(file_path, 'w') as f:
            f.write("West Wind,Low Pressure,Red Sky,Barometer,Rain\n")
            for _ in range(rows):
                west_wind = random.choice([True, False])
                low_pressure = random.choice([True, False])
                red_sky = random.random() < 0.8 if west_wind else random.random() < 0.2
                barometer = random.random() < 0.9 if low_pressure else random.random() < 0.1
                if low_pressure and west_wind:
                    rain = random.random() < 0.1
                elif not low_pressure and not west_wind:
                    rain = random.random() < 0.2
                else:
                    rain = random.random() < 0.9
                f.write(f"{west_wind},{low_pressure},{red_sky},{barometer},{rain}\n")
    data = pd.read_csv(file_path)
    #typecast to int
    data = data.astype(int)
    return data

if __name__ == "__main__":
    data = get_weather_data()
    #completare
    

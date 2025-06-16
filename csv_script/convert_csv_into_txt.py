import os
import pickle
import requests
import numpy as np
import pandas as pd

# use the all_seasons southpark dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'all_seasons.csv')
data_csv = pd.read_csv(input_file_path)

# since the original dataset was in .csv format, we also need to filter out all episode and season information
data_txt = ''.join([row['Character'] + ': ' + row['Line'] for i, row in data_csv.iterrows()])

# store as .txt
data_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(data_path, 'w', encoding='utf-8') as f:
    f.write(data_txt)
pass


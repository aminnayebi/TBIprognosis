import os
from tqdm import tqdm
import pandas as pd
import numpy as np


# Function to process any one set of data.
def process(set_name):
    
    # Data directory.
    data_dir = './data/'
    
    # Read all data into a dataframe.
    rows = []
    for fn in tqdm(os.listdir(data_dir+'set-'+set_name)):
        with open(data_dir+'set-'+set_name+'/'+fn) as f:
            lines = f.readlines()
        pat_id = int(lines[1].split(',')[2][:-1])
        for line in lines[2:]:
            line_split = line[:-1].split(',')
            hour_min = line_split[0].split(':')
            var_name = line_split[1]
            var_val = float(line_split[2])
            rows.append([pat_id, int(hour_min[0]), int(hour_min[1]), var_name, var_val])
    df = pd.DataFrame(rows, columns =['pat_id', 'hour', 'min', 'name', 'value'])
    
    # Remove observations with no variable name.
    df = df.loc[df['name']!='']
    
    # Keep observations only for first 48 hours.
    df = df.loc[df['hour']<=47]
    
    # Merge some variables.
    for v in ['DiasABP', 'MAP', 'SysABP']:
        df.replace(v, '(NI)'+v, inplace=True)
        df.replace('NI'+v, '(NI)'+v, inplace=True)
    
    # Keep only variables needed.
    rm_var = ['Age', 'Gender', 'MechVent', 'Height', 'ICUType']
    df = df.loc[~df['name'].isin(rm_var)]
    
    # Delete rows with ununsual values.
    df = df.loc[(df['name']!='Weight')|(df['value']>4)]
    df = df.loc[(df['name']!='pH')|(df['value']<=14)]
    
    # Group values hourly.
    df = df.groupby(['pat_id','hour', 'name']).agg({'value':np.mean})
    df.reset_index(inplace=True)
    
    # Add outcomes to dataframe.
    with open(data_dir+'Outcomes-'+set_name+'.txt') as f:
        lines = f.readlines()[1:]
    rows = []
    for line in lines:
        line_split = line[:-1].split(',')
        rows.append([int(line_split[0]), int(line_split[5])])
    out = pd.DataFrame(rows, columns=['pat_id', 'in_hosp_death'])
    df = df.merge(out, on='pat_id')
    
    # Save to file.
    df.to_csv(data_dir+'set-'+set_name+'.csv', index=False)
  
  
# Process the three sets of data.
process('a')
process('b')
process('c')
    

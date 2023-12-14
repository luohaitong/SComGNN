#对中频和低频的embedding重构出的adj进行分析
import pandas as pd
import numpy as np

dataset = 'Appliances'
adj_low = pd.read_csv('adj_low.csv')
adj_mid = pd.read_csv('adj_mid.csv')

path = './dataset/processed_val/' + str(dataset) + '.npz'
data = np.load(path, allow_pickle=True)

features = data['features']
com_edge_index = data['com_edge_index']
train_set = data['train_set']
val_set = data['val_set']
test_set = data['test_set']

print(features[3,:])
print(features[4,:])
print(features[5,:])
print(features[9,:])
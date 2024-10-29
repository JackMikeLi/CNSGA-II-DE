import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
w1,w2,w3 = 0.8,0.1,0.1
# w1 = w2 = w3 = 1/3
print(w1,w2,w3)
def make_decision(uncertainy_t,uncertainy_beta,G1_true,G2_true,G3_true,type):
    filename_y = '../Result/{}-{}-y_result.xlsx'.format(uncertainy_t, uncertainy_beta)
    filename_x = '../Result/{}-{}-x_result.xlsx'.format(uncertainy_t, uncertainy_beta)
    data_x = pd.read_excel(filename_x)
    data_y = pd.read_excel(filename_y)
    data_y = data_y.drop([data_y.columns[0]], axis=1)
    g1_max, g1_min = np.max(data_y[0]), np.min(data_y[0])
    g2_max, g2_min = np.max(data_y[1]), np.min(data_y[1])
    g3_max, g3_min = np.max(data_y[2]), np.min(data_y[2])
    best_index = -1
    best_value = []
    best_D = 0xffffffff

    for index, (g1, g2, g3) in enumerate(data_y.values):
        if g1 < G1_true or g2 <G2_true or g3 > G3_true:
            continue
        d = w1 * ((g1 - g1_min) / (g1_max - g1_min)) ** 2 + w2 * ((g2 - g2_min) / (g2_max - g2_min)) ** 2 + w3 * (
                    (g3_max - g3) / (g3_max - g3_min)) ** 2
        d = np.sqrt(d)
        if d < best_D:
            best_index = index
            best_value = [g1, g2, g3]
            best_D = d
    print('{},best_index:{}'.format(type[0],best_index))
    start_index = best_index * 21
    end_index = start_index + 16
    best_x = data_x.drop([data_x.columns[0]], axis=1).loc[start_index:end_index - 1].values
    return best_x,best_value
G1_true = 19670136623.873913
G2_true = 4.81265426132003
G3_true = 4.694839349506769
df_x = pd.DataFrame()
gap_size = 5
df_y = pd.DataFrame()
for i in [0,0.2,0.4,0.6,0.8]:
    for j in [0.4,0.6,0.8]:
        type = np.array(['({},{})'.format(i,j)] * 16).reshape(-1,1)
        uncertainy_t = i
        uncertainy_beta = j
        best_x,best_y = make_decision(uncertainy_t,uncertainy_beta,G1_true,G2_true,G3_true,type)
        best_x = np.concatenate((best_x,type),axis=1)
        best_y = best_y + ['({},{})'.format(i,j)]
        best_x = pd.DataFrame(best_x)
        best_y = pd.DataFrame(np.array(best_y).reshape(1,-1))
        df_x = pd.concat((df_x,best_x),axis=0)
        emyty_rows = pd.DataFrame([[None] * best_x.shape[1]] * gap_size)
        df_x = pd.concat((df_x,emyty_rows),axis=0)

        df_y = pd.concat((df_y,best_y),axis=0)
folder_path = '../Result/Analysis'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
df_x.to_excel('{}/best_x_result.xlsx'.format(folder_path))
df_y.to_excel('{}/best_y_result.xlsx'.format(folder_path))



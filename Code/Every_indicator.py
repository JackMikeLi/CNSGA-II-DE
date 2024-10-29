import os

import numpy as np
import pandas as pd
from Load_data import *

# 计算目标函数
def function(x):
    S, IW = x[:, :3] * 1e4, x[:, 3] * 1e8
    G1, Gini, Y = cal_G1(S, IW)
    G2,_,_ ,RE,CS= cal_G2(S, IW)
    G3,WF,wf_green,wf_grey,ef_all,EF = cal_G3(S, IW)
    if constraint(x) == False:
        G1,G2,G3 = -0xffffffffff,-0xffffffffff,0xffffffffff
    print('G1:',G1)
    print('G2:',G2)
    print('G3:',G3)
    return wf_green,wf_grey,ef_all,Y,Gini,G1,RE,CS,G2,WF,EF,G3
def cal_G1(S,IW):
    Y = np.sum(np.sum(S * Yic,axis=1)) # 产量，单位 kg
    Y_all = np.sum(S * Yic,axis=1)
    k = 1/(2 * citiesnum * np.sum(IW * (1-p_loss)))
    g = 0
    for i in range(0,citiesnum):
        p1 = p_loss[i]
        iw1 = IW[i]
        for j in range(0,citiesnum):
            p2 = p_loss[j]
            iw2 = IW[j]
            g = g + np.abs(iw1*(1 - p1) - iw2*(1 - p2))

    Gini = k * g # 水基尼系数
    G1 = (1 - Gini) * Y
    return G1,Gini,Y_all
def cal_G2(S,IW):
    AEB = []
    EB = []
    RE,CS = [],[]
    for i in range(0,citiesnum):
        s, iw = S[i, :], IW[i]
        yic = Yic[i,:]
        p = p_loss[i]

        up = []
        r = 0
        c = 0
        for j in range(0,cropsnum):
            # 计算 RE
            re = s[j] * yic[j] * Pc[j] / 1e3 # 单位:yuan
            r = r + re
            # 计算 CS
            cs = np.sum(K_m[:,j] * s[j]) # 单位: yuan
            c = c + cs
            up.append(re - cs)
        RE.append(r)
        CS.append(c)
        EB.append(np.mean(up) / np.max(up))
        up = np.sum(up)
        down = (1 - p) * iw
        AEB.append(up/down) # 单位：yuan/m^3
    G2 = np.mean(AEB) # 单位: yuan/ m^3
    return G2,AEB,EB,RE,CS
def cal_G3(S,IW):
    c_max,c_nat = 0.02,0
    alpha = 0.1
    WF,EF = [],[]
    wf_green,wf_grey = [],[]
    ef_all = []
    for i in range(0,citiesnum):
        s,iw = S[i,:],IW[i]
        fer = q_fer[i]

        # 计算WF
        WF_green,WF_grey = 0,0
        for j in range(0,cropsnum):
            # 计算WF_green
            etc = ETc[j]
            pe = Pe[j]
            day = int(lgp[j])
            # for k in range(0,day):
            #     WF_green = WF_green + min(etc,pe) * s[j]
            WF_green = WF_green + min(etc,pe) * s[j] * day

            # 计算WF_grey
            WF_grey = WF_grey + (alpha * fer * s[j]) / (c_max - c_nat)

        wf_green.append(WF_green)
        wf_grey.append(WF_grey)
        WF.append(iw + WF_green + WF_grey)
        # 计算EF
        ef = 0
        q_n = Q_n[i,:]
        tem = []
        for j in range(0,cropsnum):
            # for k in range(N_sourcesnum):
            #     ef = ef + q_n[k] * H_n[k] * s[j]
            ef = ef + np.sum(q_n * H_n * s[j])
            tem.append(q_n * H_n * s[j])
        ef_all.append(tem)
        EF.append(ef)
    WF = np.array(WF) / 1e8 # TODO 调整单位 10^8 m^3
    EF = np.array(EF) / 1e9 # TODO 调整单位 10^9 MJ
    # 标准化
    WF_tem = (WF - np.min(WF)) / (np.max(WF) - np.min(WF))
    EF_tem = (EF - np.min(EF)) / (np.max(EF) - np.min(EF))

    G3 = np.sum(WF_tem * EF_tem)
    return G3,WF,wf_green,wf_grey,ef_all,EF

def constraint(x):
    S, IW = x[:, :3] * 1e4, x[:, 3] * 1e8# S 的单位是 ha,IW 的单位是 m^3
    G2,AEB,EB,RE,CS = cal_G2(S,IW)
    # 约束一： 水资源可用性约束
    strain1_left = np.sum(IW)
    strain1_right = np.sum((SW + GW + OW - EW - FW - DW) * 1e8)
    if strain1_left > strain1_right:
        print('水资源可用性约束不满足', strain1_left, strain1_right)
        return False
    # 约束二：  经济损失风险约束
    strain2_left = 1 - np.mean(((np.array(EB) + 0.3)* IW / ((SW + GW + OW - EW - FW - DW) * 1e8)))
    strain2_right = uncertainy_beta
    if strain2_left > strain2_right:
        print('经济损失风险约束不满足',strain2_left,strain2_right)
        return False

    # 约束五： 电能供应约束
    strain5_left = np.sum(np.sum(S * Yic, axis=1))
    strain5_right = np.sum(PCF * POP)
    if strain5_left < strain5_right:
        print('电能供应约束不满足')
        return False

    for i in range(0,citiesnum):
        s,iw = S[i,:],IW[i]

        # 约束二：  经济损失风险约束
        sw1, sw2, ew, dw, fw, gw, ow = SW1[i], SW2[i], EW[i], DW[i], FW[i], GW[i], OW[i]
        sw = SW[i]
        aeb = AEB[i]

        # 约束三：  灌溉需水量约束
        paw, pre = PAW[i], PRE[i]
        strain3_left = np.sum(s) * paw * pre
        strain3_right = (1 - p_loss[i]) * iw
        if strain3_left > strain3_right:
            print('灌溉需水量约束不满足')
            return False

        # 约束四： 电能供应约束
        strain4_left = np.sum(s) * q_far[i]
        strain4_right = EPmax[i]
        if strain4_left > strain4_right:
            print('电能供应约束不满足')
            return False
    return True

def get_x(uncertainy_t,uncertainy_beta,type):
    filename_y = '../Result/Analysis/best_y_result.xlsx'
    filename_x = '../Result/Analysis/best_x_result.xlsx'
    print(filename_x)
    data_x = pd.read_excel(filename_x)
    data_y = pd.read_excel(filename_y)
    x = data_x.loc[data_x[4] == type].drop([data_x.columns[0]], axis=1).values
    y = data_y.loc[data_y[3] == type].drop([data_y.columns[0]], axis=1).values
    return x
def cal_all_indicator(x):
    wf_green, wf_grey, ef_all, Y, Gini, G1, RE, CS, G2, WF, EF, G3 = function(x)
    wf_green = np.array(wf_green).reshape(-1,1)
    wf_grey = np.array(wf_grey).reshape(-1,1)

    ef = np.array(ef_all)
    ef_all = np.sum(ef,axis=1)
    Y = np.array(Y).reshape(-1,1)
    Gini = np.array([Gini] * 16).reshape(-1,1)
    G1 = np.array([G1] * 16).reshape(-1,1)
    RE = np.array(RE).reshape(-1,1)
    CS = np.array(CS).reshape(-1,1)
    G2 = np.array([G2] * 16).reshape(-1, 1)
    WF = np.array(WF).reshape(-1,1)
    EF = np.array(EF).reshape(-1,1)
    G3 = np.array([G3] * 16).reshape(-1, 1)

    df = np.concatenate((wf_green, wf_grey, ef_all, Y, Gini, G1, RE, CS, G2, WF, EF, G3),axis=1)
    columns = ['wf_green','wf_grey','EF1','EF2','EF3','EF4','EF5','Y','Gini','G1','RE','CS','G2','WF','EF','G3']
    df = pd.DataFrame(df,columns=columns)
    return df
# Target
uncertainy_t = 0.4
uncertainy_beta = 0.4
type = '({},{})'.format(uncertainy_t, uncertainy_beta)

x = get_x(uncertainy_t,uncertainy_beta,type)
x_true = dataS7.drop(['Item'],axis=1).values
df1 = cal_all_indicator(x)
df2 = cal_all_indicator(x_true)

folder_path = '../Result/Analysis'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

filename1 = folder_path + '/' + type + '-result.xlsx'
df1.to_excel(filename1)
filename1 =  '{}/现状-result.xlsx'.format(folder_path)
df2.to_excel(filename1)


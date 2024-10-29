import pandas as pd
import numpy as np
nameDict = {1:'Chongqing',2:'Chengdu',3:'Zigong',4:'Luzhou',5:'Deyang',6:'Mianyang',7:'Suining',8:'Neijiang',
            9:'Leshan',10:'Nanchong',11:'Meishan',12:'Yibin',13:"Guang'an",14:'Dazhou',15:"Ya'an",16:'Ziyang'}
citiesnum,cropsnum,N_sourcesnum,M_costnum = 16,3,5,6
# 读文件
dataS1 = pd.read_excel('../Data/TableS1.xlsx') # 各地区的水资源相关数据
dataS2 = pd.read_excel('../Data/TableS2.xlsx') # 各地区作物产量的相关数据
dataS3 = pd.read_excel('../Data/TableS3.xlsx') # 农业生产中各资源的投入成本
dataS4 = pd.read_excel('../Data/TableS4.xlsx') # 农业生产中资源消耗的相关数据
dataS5 = pd.read_excel('../Data/TableS5.xlsx') # 各地区的土地政策
dataS6 = pd.read_excel('../Data/TableS6.xlsx') # 各地区的其他相关数据
dataS7 = pd.read_excel('../Data/TableS7.xlsx') # 2022年各地区的实际种植面积及农业用水量相关数据
dataS8 = pd.read_excel('../Data/TableS8.xlsx') # 其他数据
uncertainy_t = 0.2
uncertainy_beta = 0.8
# 提取数据
S1 = dataS1.drop(['Water'],axis=1).values
SW1,SW2,EW,DW,FW,GW,OW = S1[:,0],S1[:,1],S1[:,2],S1[:,3],S1[:,4],S1[:,5],S1[:,6]
# SW = np.random.uniform(SW1 - SW2 * uncertainy_t,SW1 + SW2 * uncertainy_t)
SW = SW1 + SW2 * uncertainy_t
IW_max = SW + GW + OW - EW - FW - DW

S2 = dataS2[['Rice','Wheat','Corn']].values
Yic = S2[:16,:] # 分区i作物c的单位面积产量（kg/ha）
Pc = S2[16,:] # 作物c单位产量收入（元/10^3kg）

K_m = dataS3.drop(['Resource input cost (yuan/ha)'],axis=1).values # 第m种资源单位面积成本（元/公顷）

q_fer = dataS4['Fertilizer'][:16].values # 单位:(kg/ha)
q_far = dataS4['Electricity'][:16].values # 单位:(kwh/ha)
Q_n = dataS4.drop(['Item'],axis=1).iloc[:16].values # 单位面积第n种资源利用率（kg/ha）
H_n = dataS4.drop(['Item'],axis=1).loc[16].values # 第n种资源的能耗系数

S_limit = dataS5.drop(['Item'],axis=1).values  # 单位: ha
bound_min = np.concatenate((S_limit[:,[1,3,5]],np.zeros(citiesnum).reshape(-1,1)),axis=1)
bound_max = np.concatenate((S_limit[:,[0,2,4]],IW_max.reshape(-1,1)),axis=1)
S_max = S_limit[:,[0,2,4]]
p_loss = dataS6['p,loss'].values # 分区i的失水率
PRE = dataS6['pre'].values # 单位：ha/ha
POP = dataS6['pop'].values * 1e4 # 单位: person
PAW = dataS6['PAW'].values * 1e4 # 单位: m^3 / ha
PCF = dataS6['PCF'].values # 单位: kg/person
EPmax = dataS6['EP,max'].values * 1e4 # 单位: kwh
ETc = dataS8[['Rice','Wheat','Corn']].iloc[0].values # 单位 :m^3/(day * ha)
Pe = dataS8[['Rice','Wheat','Corn']].iloc[1].values # 单位 :m^3/(day * ha)
lgp = dataS8[['Rice','Wheat','Corn']].iloc[2].values # 单位 ：/day

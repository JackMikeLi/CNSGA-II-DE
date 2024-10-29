from matplotlib import pyplot as plt
from Load_data import *

# 计算目标函数
def function(x):
    print('现状:')
    S, IW = x[:, :3] * 1e4, x[:, 3] * 1e8
    G1 = cal_G1(S, IW)
    G2,_,_ = cal_G2(S, IW)
    G3 = cal_G3(S, IW)
    if constraint(x) == False:
        G1,G2,G3 = -0xffffffffff,-0xffffffffff,0xffffffffff
    print('G1:',G1)
    print('G2:',G2)
    print('G3:',G3)
    return G1,G2,G3
def cal_G1(S,IW):
    Y = np.sum(np.sum(S * Yic,axis=1)) # 产量，单位 kg
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
    return G1
def cal_G2(S,IW):
    AEB = []
    EB = []
    for i in range(0,citiesnum):
        s, iw = S[i, :], IW[i]
        yic = Yic[i,:]
        p = p_loss[i]

        up = []
        for j in range(0,cropsnum):
            # 计算 RE
            re = s[j] * yic[j] * Pc[j] / 1e3# 单位:yuan
            # 计算 CS
            cs = np.sum(K_m[:,j] * s[j]) # 单位: yuan
            up.append(re - cs)
        EB.append(np.mean(up) / np.max(up))
        up = np.sum(up)
        down = (1 - p) * iw
        AEB.append(up/down) # 单位：yuan/m^3
    G2 = np.mean(AEB) # 单位: yuan/ m^3
    return G2,AEB,EB
def cal_G3(S,IW):
    c_max,c_nat = 0.02,0
    alpha = 0.1
    WF,EF = [],[]
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
        WF.append(iw + WF_green + WF_grey)
        # 计算EF
        ef = 0
        q_n = Q_n[i,:]
        for j in range(0,cropsnum):
            # for k in range(N_sourcesnum):
            #     ef = ef + q_n[k] * H_n[k] * s[j]
            ef = ef + np.sum(q_n * H_n * s[j])
        EF.append(ef)
    WF = (WF - np.min(WF)) / (np.max(WF) - np.min(WF))
    EF = (EF - np.min(EF)) / (np.max(EF) - np.min(EF))

    G3 = np.sum(np.power(WF * EF, 1 / 3))
    return G3

def constraint(x):
    S, IW = x[:, :3] * 1e4, x[:, 3] * 1e8# S 的单位是 ha,IW 的单位是 m^3
    _,AEB,EB = cal_G2(S,IW)
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
def generate_x(matrix,std_dev):
    new_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        perturbation = np.random.normal(loc=0.0, scale=std_dev[i], size=matrix.shape[1])
        new_matrix[i] = matrix[i] + perturbation
    return new_matrix
def plot_compare():
    # 用实际x进行测试计算
    x = dataS7.drop(['Item'],axis=1)
    x = x.values
    G = function(x)
    data = pd.read_excel('y_result.xlsx')
    data = data.drop('Unnamed: 0',axis=1).values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y,Z = data[:,0],data[:,1],data[:,2]
    x,y,z = G[0],G[1],G[2]
    ax.scatter(X,Y,Z,c = 'b')
    ax.scatter(x,y,z,c = 'r')
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_zlabel('F3')
    plt.show()
x_true = dataS7.drop(['Item'],axis=1).values
df2 = function(x_true)

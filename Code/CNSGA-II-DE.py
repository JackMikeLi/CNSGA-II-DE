import os
import matplotlib.pyplot as plt
import random
from Load_data import *
from collections import defaultdict

class Individual:
    def __init__(self):
        self.solution = None
        self.objective = defaultdict()

        self.n = 0
        self.rank = 0
        self.S = []
        self.distance = 0

    def bound_process(self,bound_min,bound_max):
        for i in range(0,citiesnum):
            for j in range(0,cropsnum):
                x = self.solution[i][j]
                if x > bound_max[i][j]:
                    self.solution[i][j] = bound_max[i][j]
                elif x < bound_min[i][j]:
                    self.solution[i][j] = bound_min[i][j]

    def cal_objective(self,objective_fun):
        self.objective = objective_fun(self.solution)

    def __lt__(self,other):
        v1 = list(self.objective.values())
        v2 = list(other.objective.values())
        if v1[0] < v2[0] or v1[1] < v2[1] or v1[2] > v2[2]:
            return 0
        return 1



def generate_x(matrix,std_dev):
    new_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        perturbation = np.random.normal(loc=0.0, scale=std_dev[i], size=matrix.shape[1])
        new_matrix[i] = matrix[i] + perturbation
    return new_matrix
def init_solution(num):
    # 用实际x进行测试计算
    x = dataS7.drop(['Item'], axis=1).values # 现状解
    all_x = []
    all_x.append(x)
    std_dev = np.ones(x.shape[0])
    current = x.copy()

    equl_x = np.linspace(bound_min,bound_max,num)
    for x_tem in equl_x:
        if constraint(x_tem):
            all_x.append(x_tem)

    while len(all_x) < num:
        x_current = np.clip(generate_x(current, std_dev), bound_min, bound_max)
        x_current_2 = np.random.uniform(bound_min,bound_max)
        if constraint(x_current) == True:
            print('可行解', x_current)
            all_x.append(x_current)
            std_dev = np.std(x_current, axis=1)
            current = x_current
        if constraint(x_current_2) == True:
            print('可行解', x_current_2)
            all_x.append(x_current_2)
            current = x_current
        print('size:', len(all_x))
    return np.array(all_x)

def run():
    generations = 200
    popnum = 100
    eta = 0.5

    poplength = len(bound_min)
    objective_fun = Function

    # 生成第一代种群
    P = []
    all_x = init_solution(popnum)
    for i in range(popnum):
        P.append(Individual())
        # P[i].solution = np.random.uniform(bound_min,bound_max)
        P[i].solution = all_x[i]
        P[i].bound_process(bound_min,bound_max)
        P[i].cal_objective(objective_fun)
    fast_non_dominated_sort(P)
    # 生成子代种群Q
    Q = make_new_pop(P, eta, bound_min, bound_max, objective_fun)
    fast_non_dominated_sort(P)
    P_t = P # 当前的父代
    Q_t = Q # 当前的子代

    for gen_cur in range(1,generations+1):
        R_t = P_t + Q_t # 将当前父代和子代合并成一个种群
        F = fast_non_dominated_sort(R_t)

        P_n = [] # 即P_t+1,表示下一代的父代
        i = 1
        while len(P_n) + len(F[i]) < popnum:
            crowding_distance_assignment(F[i])
            P_n = P_n + F[i]
            i = i + 1
        F[i].sort(key=lambda x: x.distance)
        P_n = P_n + F[i][:popnum - len(P_n)]  # 将排序后的个体加入下一代父代种群，以保证种群大小不超过 popnum
        Q_n = make_new_pop(P_n, eta, bound_min, bound_max,
                           objective_fun)  # 使用选择、交叉和变异操作创建新的子代种群 Q_n。

        P_t = P_n
        Q_t = Q_n

        # 绘图
        if gen_cur % 10 == 0:
            plt.clf()
            plot_P(P_t,gen_cur)
            plt.pause(0.1)
    plt.show()
    # 提取所有解
    R_t = P_t + Q_t  # 将当前父代和子代合并成一个种群
    all_x,all_y= [],[]
    F = fast_non_dominated_sort(R_t)
    for f in F[1]:
        x,y = f.solution,list(f.objective.values())
        all_x.append(x)
        all_y.append(y)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    print(all_x.shape,all_y.shape)
    df1 = pd.DataFrame()
    gap_size = 5
    for i in range(all_x.shape[0]):
        x  = pd.DataFrame(all_x[i])
        df1 = pd.concat((df1,x),axis=0)
        empty_rows = pd.DataFrame([[None] * x.shape[1]] * gap_size)
        df1 = pd.concat((df1,empty_rows),axis=0)
    folder_path = '../Result'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    type = str(uncertainy_t) + '-' + str(uncertainy_beta)
    df1.to_excel('{}/{}-x_result.xlsx'.format(folder_path,type))

    df2 = pd.DataFrame(all_y)
    df2.to_excel('{}/{}-y_result.xlsx'.format(folder_path,type))

    return 0

def fast_non_dominated_sort(P):
    """
    非支配排序
    :param P: 种群 P
    :return F: F=(F_1, F_2, ...) 将种群 P 分为了不同的层， 返回值类型是dict，键为层号，值为 List 类型，存放着该层的个体
    """
    F = defaultdict(list)  # F 一个字典,键为非支配层的层号rank,值为一个列表--存储该层的个体

    for p in P:
        p.S = []  # 支配的集合
        p.n = 0  # 被支配的个数
        # 对于每个个体 p，遍历种群中的其他个体 q，根据 p 与 q 的支配关系，更新 p 的支配集合 S 和被支配计数器 n。
        for q in P:
            if p < q:  # if p dominate q
                p.S.append(q)  # Add q to the set of solutions dominated by p
            elif q < p:
                p.n += 1  # Increment the domination counter of p
        # 如果 n 为零，则表示 p 不被任何个体支配，将其分配到第一层 F[1] 中
        if p.n == 0:
            p.rank = 1
            F[1].append(p)
    # 接下来，对于每一层 F[i]，更新被当前层个体支配的个体被支配计数器，并将新的非支配个体添加到下一层 F[i+1]。
    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i = i + 1
        F[i] = Q

    return F

def make_new_pop(P, eta, bound_min, bound_max, objective_fun):
    """
    use select,crossover and mutation to create a new population Q
    :param P: 父代种群
    :param eta: 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
    :param bound_min: 定义域下限
    :param bound_max: 定义域上限
    :param objective_fun: 目标函数
    :return Q : 子代种群
    """
    popnum = len(P)
    Q = []
    # binary tournament selection 二元锦标赛选择父代
    for i in range(int(popnum / 2)):  # 让父代种群个数为双数 能避免不必要的麻烦
        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent1
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent1 = binary_tournament(P[i], P[j])

        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent2
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent2 = binary_tournament(P[i], P[j])

        while (parent1.solution == parent2.solution).all():  # 如果选择到的两个父代完全一样，则重选另一个
            # 上面的.all可能导致警告,因为如果 parent1.solution不是数组，而是单个布尔值，就不应该使用 .all() 方法
            # 但是我们知道解向量是单个布尔值，所以可以安全地忽略这个警告。
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = binary_tournament(P[i], P[j])

        # parent1 和 parent1 进行交叉，变异 产生 2 个子代
        Two_offspring = crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun)

        # 产生的子代进入子代种群
        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    return Q
def crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun):
    """
    交叉方式使用二进制交叉算子（SBX），变异方式采用多项式变异（PM）
    :param parent1: 父代1
    :param parent2: 父代2
    :param eta: 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
    :param bound_min: 定义域下限
    :param bound_max: 定义域上限
    :param objective_fun: 目标函数
    :return: 2 个子代
    """
    poprow,popcol = parent1.solution.shape # 获得解向量的维度或者分量个数,表示解向量的维度

    offspring1 = Individual()
    offspring2 = Individual()
    # 创建一个指定长度的一维数组，该数组的元素值是未初始化的，即它们可能包含任意值。这个数组将被用来存储新生成的子代个体的解向量。
    # np.empty 仅仅分配了内存空间而不初始化数组元素的值
    offspring1.solution = np.empty((poprow,popcol))
    offspring2.solution = np.empty((poprow,popcol))

    # 二进制交叉
    for i in range(0,poprow):
        for j in range(0,popcol):
            rand = random.random()  # 生成一个在 [0, 1) 范围内的随机数
            #  beta的计算采用了 SBX 的公式，其中 eta 是变异分布参数，用于调整交叉的程度。
            beta = (rand * 2) ** (1 / (eta + 1)) if rand < 0.5 else (1 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
            offspring1.solution[i][j] = 0.5 * ((1 + beta) * parent1.solution[i][j] + (1 - beta) * parent2.solution[i][j])
            offspring2.solution[i][j] = 0.5 * ((1 - beta) * parent1.solution[i][j] + (1 + beta) * parent2.solution[i][j])

    # 多项式变异
    # TODO 变异的时候只变异一个，不要两个都变，不然要么出现早熟现象，要么收敛速度巨慢 why？
    # 通过只变异一个个体，可以确保种群中的每个个体都有机会经历一些变化，而不会太快地趋向于某个方向。这有助于维持多样性，促使算法更好地探索搜索空间。
    # 另一方面，如果每次都同时变异两个个体，可能导致整个种群在搜索空间中以较大的步伐移动，这可能使得算法更容易陷入局部最优解，而不太容易跳出这些局部最优解。
    for i in range(0,poprow):
        for j in range(0,popcol):
            mu = random.random()  # 生成一个在 [0, 1) 范围内的随机数
            # delta 的计算采用了 PM 的公式，其中 eta 是变异分布参数，用于调整变异的程度。
            delta = 2 * mu ** (1 / (eta + 1)) if mu < 0.5 else (1 - (2 * (1 - mu)) ** (1 / (eta + 1)))
            offspring1.solution[i][j] = offspring1.solution[i][j] + delta


    # 定义域越界处理
    offspring1.bound_process(bound_min, bound_max)
    offspring2.bound_process(bound_min, bound_max)

    # 计算目标函数值
    offspring1.cal_objective(objective_fun)
    offspring2.cal_objective(objective_fun)

    return [offspring1, offspring2]

def binary_tournament(ind1, ind2):
    """
    二元锦标赛
    :param ind1:个体1号
    :param ind2: 个体2号
    :return:返回较优的个体
    """
    if ind1.rank != ind2.rank:  # 如果两个个体有支配关系，即在两个不同的rank中，选择rank小的
        return ind1 if ind1.rank < ind2.rank else ind2
    elif ind1.distance != ind2.distance:  # 如果两个个体rank相同，比较拥挤度距离，选择拥挤读距离大的
        # 如果是初代父种群P生成第一子代时,此时的每一个解决方案个体还没有distance的赋值,默认都是0
        return ind1 if ind1.distance > ind2.distance else ind2
    else:  # 如果rank和拥挤度都相同，返回任意一个都可以
        return ind1

def crowding_distance_assignment(L):
    """ 传进来的参数应该是L = F(i)，类型是List,传进来的一层rank的一个list集合"""
    l = len(L)  # number of solution in F F层共有多少个个体

    for i in range(l):
        L[i].distance = 0  # initialize distance 初始化所有个体的拥挤度为0

    for m in L[0].objective.keys():  # 对每个目标距离进行拥挤度距离计算
        L.sort(key=lambda x: x.objective[m])  # sort using each objective value根据当前目标方向值对个体进行排序。
        # 将第一个和最后一个个体的拥挤度距离设为无穷大，确保它们总是被选择。
        L[0].distance = float('inf')
        L[l - 1].distance = float('inf')  # so that boundary points are always selected

        # 排序是由小到大的，所以最大值和最小值分别是 L[l-1] 和 L[0]
        f_max = L[l - 1].objective[m]
        f_min = L[0].objective[m]

        # 当某一个目标方向上的最大值和最小值相同时，此时会发生除零错，这里采用异常处理机制来解决
        try:
            for i in range(1, l - 1):  # for all other points
                L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
        except Exception:
            print(str(m) + "目标方向上，最大值为" + str(f_max) + "最小值为" + str(f_min))
def plot_P(P,gen_cur):
    """
    假设目标就俩,给个种群绘图
    :param P:
    :return:
    """
    X = []
    Y = []
    Z = []
    for ind in P:
        X.append(ind.objective[1])
        Y.append(ind.objective[2])
        Z.append(ind.objective[3])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X,Y,Z)
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_zlabel('F3')
    ax.set_title('current generation_P_t:' + str(gen_cur))  # 绘制当前循环选出来的父代的图

# 计算目标函数

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
            re = s[j] * yic[j] * Pc[j] / 1e3 # 单位:yuan
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
            ef = ef + np.sum(q_n * H_n * s[j])
        EF.append(ef)
    WF = (WF - np.min(WF)) / (np.max(WF) - np.min(WF))
    EF = (EF - np.min(EF)) / (np.max(EF) - np.min(EF))

    G3 = np.sum(np.power(WF * EF, 1 / 3))
    return G3
def constraint(x):
    S, IW = x[:, :3] * 1e4, x[:, 3] * 1e8# S 的单位是 ha,IW 的单位是 m^3
    _,AEB,EB = cal_G2(S, IW)
    # 约束一： 水资源可用性约束
    strain1_left = np.sum(IW)
    strain1_right = np.sum((SW + GW + OW - EW - FW - DW) * 1e8)
    if strain1_left > strain1_right:
        print('水资源可用性约束不满足', strain1_left, strain1_right)
        return False
    # 约束二：  经济损失风险约束
    EB = np.array(EB)
    if uncertainy_beta == 0.4:
        EB = EB + 0.1
    strain2_left = 1 - np.mean((EB * IW / ((SW + GW + OW - EW - FW - DW) * 1e8)))
    strain2_right = uncertainy_beta
    if strain2_left > strain2_right:
        print('经济损失风险约束不满足',strain2_left,strain2_right)
        return False

    # 约束五： 电能供应约束
    strain5_left = np.sum(np.sum(S * Yic, axis=1))
    strain5_right = np.sum(PCF * POP)
    if strain5_left < strain5_right:
        print('电能供应约束不满足',strain5_left,strain5_right)
        return False

    for i in range(0,citiesnum):
        s,iw = S[i,:],IW[i]


        # 约束三：  灌溉需水量约束
        paw, pre = PAW[i], PRE[i]
        strain3_left = np.sum(s) * paw * pre
        strain3_right = (1 - p_loss[i]) * iw
        if strain3_left > strain3_right:
            print('灌溉需水量约束不满足',strain3_left,strain3_right)
            return False

        # 约束四： 电能供应约束
        strain4_left = np.sum(s) * q_far[i]
        strain4_right = EPmax[i]
        if strain4_left > strain4_right:
            print('电能供应约束不满足',strain4_left,strain4_right)
            return False
    return True
def Function(x):
    S, IW = x[:, :3]* 1e4, x[:, 3]* 1e8
    if constraint(x):
        G1 = cal_G1(S, IW)
        G2,_,_ = cal_G2(S, IW)
        G3 = cal_G3(S, IW)
    else:
        G1,G2,G3 = -0xffffffffff,-0xffffffffff,0xffffffffff
    f = defaultdict(float)  # 定义这个字典f，存放后续的两个目标函数值
    f[1] = G1
    f[2] = G2
    f[3] = G3
    return f
def run_all():
    global uncertainy_beta
    global uncertainy_t
    for i in [0,0.2,0.4,0.6,0.8]:
        for j in [0.4,0.6,0.8]:
            uncertainy_t = i
            uncertainy_beta = j
            run()
run_all()

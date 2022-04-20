import math
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from tasks import Tasks
import random
avg_list = []
total_list = []
count_list =[]

avg_each_tasks=[]
summ = 0

def plot_avgtasks(avg):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Test episdoes")
    host.set_ylabel("Average profit")

    plt.title('genetic algorithm')

    # plot curves
    #p1, = host.plot(range(len(profit)), profit, label="Priority Heuristic Algorithm", color = '#90EE90')
    #p2, = host.plot(range(len(avg)), avg, label="First Come First Service",color = '#87CEEB')
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    p1, = host.plot(range(len(avg)), avg, label="average profit of each task")
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
   # host.axis["left"].label.set_color(p1.get_color())
   # host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0, 10])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()

def plot_profit(profit, avg):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Number of iterations")
    host.set_ylabel("Total reward")

    plt.title('genetic algorithm')

    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit", color ='#90EE90')
    p2, = host.plot(range(len(avg)), avg, label="Average profit")
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0, 200])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()

def plot_tasks(tasks):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")

    plt.title('50 tasks in 1000s')
    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepetd Tasks")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0, 50])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()

def init(popsize, n):
    population = []
    for i in range(popsize):
        pop = ''
        for j in range(n):
            ###每个种群内的每个个体可以取值为0或1，0为不被选择，1为被选择
            pop = pop + str(np.random.randint(0, 2))
        population.append(pop)
    return population

###交叉
##这个函数的思想就是，在每个种群的内部寻找两个共同断点，如果随机数小于给定的交叉概率就进行交叉，即每个种群分为三部分，两个断点之间的部分进行交换
##如果随机数大于交叉概率，则种群里的内容不变
##该函数的返回值为一个新的二维数组种群
def crossover(population_new, pc, ncross):
    a = int(len(population_new) / 2)
    ###选择出所有种群的双亲（所有染色体的双亲）
    parents_one = population_new[:a]
    parents_two = population_new[a:]
    ###随机每个种群（染色体的顺序）
    np.random.shuffle(parents_one)#打乱列表顺序
    np.random.shuffle(parents_two)
    ###两个父代交叉产生两个后代，假如父代分别为 abc和def 则两个后代为aec和dbf
    # 在0到1内随机生成一个实数
    ###在每个种群中产生两个断点
    # a是15，一共进行十五次
    ###后代
    offspring = []
    for i in range(a):
        r = np.random.uniform(0, 1)
        if r <= pc:

            point1 = np.random.randint(0, (len(parents_one[i]) - 1))
            point2 = np.random.randint(point1, len(parents_one[i]))

            off_one = parents_one[i][:point1] + parents_two[i][point1:point2] + parents_one[i][point2:]
            off_two = parents_two[i][:point1] + parents_one[i][point1:point2] + parents_two[i][point2:]
            ncross = ncross + 1
        else:
            off_one = parents_one[i]
            off_two = parents_two[i]
        offspring.append(off_one)
        offspring.append(off_two)
    return offspring
#返回一个交叉过后的数组，依旧是30组，依旧每组50个由0，1构成的字符串


# 对每条染色体上的每个点进行变异概率检验
##将交叉后的种群进行变异，如果随机数小于给定的变异概率就进行变异，否则不变。
##1.对于每个种群的第一个个体如果是‘1’，就变成‘0’，如果是‘0’就变成‘1’
##2.对于每个种群除第一个以外的其他个体，如果是‘1’，就变成‘0’，如果是‘0’就变成‘1’
##该函数返回一个变异后的新的population
def mutation2(offspring, pm, nmut):
    for i in range(len(offspring)):#30次
        for j in range(len(offspring[i])):#50次
            r = np.random.uniform(0, 1)
            if r <= pm:
                if j == 0:
                    if offspring[i][j] == '1':
                        offspring[i] = '0' + offspring[i][1:]
                    else:
                        offspring[i] = '1' + offspring[i][1:]
                else:
                    if offspring[i][j] == '1':
                        offspring[i] = offspring[i][:(j - 1)] + '0' + offspring[i][j:]
                    else:
                        offspring[i] = offspring[i][:(j - 1)] + '1' + offspring[i][j:]
                nmut = nmut + 1
    return offspring#返回一个和之前一样的变异后的数组

###轮盘模型
###以每个种群的价值占总价值和的比作为轮盘的构成，价值高的则占轮盘的面积大，即该染色体生存或选择概率更大
def roulettewheel(population, value, pop_num):
    fitness_sum = []
    ###价值总和
    value_sum = sum(value)
    ###每个价值的分别占比， 总和为1
    fitness = [i / value_sum for i in value]#六十个百分比
    ###从种群索引0开始逐渐构成一个总和为1的轮盘
    for i in range(len(population)):  ##60次
        if i == 0:
            fitness_sum.append(fitness[i])
        else:
            fitness_sum.append(fitness_sum[i - 1] + fitness[i])
    population_new = []
    for j in range(pop_num):

        r = np.random.uniform(0, 1)

        for i in range(len(fitness_sum)):
            if i == 0:
                if 0 <= r <= fitness_sum[i]:
                    population_new.append(population[i])
            else:
                if fitness_sum[i - 1] <= r <= fitness_sum[i]:
                    population_new.append(population[i])
    return population_new


##解码
##对于每个种群里的‘1’，代表的是可以接受，那么我们就要进行一些判断
##如果当前的总重量加上该物品的重量小于等于存储的最大重量，当前该时间段内没有任务并且到达时间加上该任务花费时间小于总时间时，任务可以接受，
##这时当前总重量就要加上该任务的重量，当前利润就要加上该任务的利润，任务列表要添加上这个任务且在该任务的执行时间内，时间窗口变为1，即在该段时间内不接受其他任务
##该函数的返回值是但前总利润和当前接受的任务的索引
def decode1(x, n, w, c, W, ttime, T,arrival_time,end_time):
    s = []  # 储存被选择物体的下标集合
    ttt = []

    ###初始化时间
    for j in range(T + 1):
        ttt.append(0)
    g = 0
    f = 0
    # for i in range(n):
    #     if x[i] == '1':
    #         ###当容量小于最大容量时，才可以继续被选择，如果超出最大容量则直接停止
    #         tt = np.random.randint(0, T)  ###选择一个该任务的开始时间
    #         count = 0
    #         if g + w <= W:
    #             for k in range(tt, tt + ttime):
    #                 if k > T - ttime:
    #                     break
    #                 elif ttt[k] == 0:
    #                     count += 1
    #             if count == ttime:
    #                 g = g + w  ###容量
    #                 f = f + c[i]  ###价值
    #                 s.append(i)
    #                 for k in range(tt, tt + ttime):
    #                     ttt[k] = 1
    #             else:
    #                 continue
    #         else:
    #             break
    # arrival_time = produce_arrive_tasks()#返回一个由50个时间构成，从小到大，范围在0到T之间的列表，到达时间列表
    # exetime = produce_exetime()#返回一个由50个10到30之间整数构成的列表，执行时间列表
    current_weight = 0

    for i in range(n):
        p = 0
        if x[i] == '1':
            if current_weight + w[i] <= W and ttt[int(arrival_time[i])] == 0 and math.ceil(end_time[i]) < T:
                for j in range(int(arrival_time[i]), math.ceil(end_time[i])):
                    if ttt[j] == 1:
                        p=1
                        break
                if p==1:
                    continue
                else:
                    current_weight += w[i]
                    f += c[i]
                    s.append(i)
                    for j in range(int(arrival_time[i]), math.ceil(end_time[i])):
                        ttt[j] = 1
            else:
                continue
    return f, s



# ##该函数通过调用decode1函数将str种群解码，每次将一个种群传给decode1函数，最后的返回值为存储每个种群利润的数组value和存储每个种群被选择的索引的二维数组
def fitnessfun1(population, n, W, ttime, T):
    value = []  ###储存每个种群的价值
    ss = []  ###储存每个种群被选择的索引
    for i in range(len(population)):
        tasks = make_tasks()
        w=[]#任务所占内存
        c=[]#任务收益
        arr_time=[]#到达时间
        end_time=[]#结束时间
        for p in range(n):
            w.append(tasks[p][4])
            c.append(tasks[p][3])
            arr_time.append(tasks[p][6])
            end_time.append(tasks[p][7])
    #return w,c,arr_time,end_time
        [f, s] = decode1(population[i], n, w, c, W, ttime, T,arr_time,end_time)#返回的是这一组的利润以及接受任务的下标
        #一共进行30次，每次对一组进行处理，一组是50个0，1
        value.append(f)
        ss.append(s)
    return value, ss
#value里一共有三十个元素，每个元素是一组50个0，1得出的利润
#ss是个二维数组，一共有三十个小组，每个小组里面是接受的每组任务的下标

def make_tasks():
    target = Tasks()
    tasks_one = target.produce_tasks()
    pick_tasks = random.sample(range(1, 151), 50)
    tasks=[]
    for i in range(len(tasks_one)):
        if tasks_one[i][0] in pick_tasks:
            tasks.append(tasks_one[i])

    rep=[]
        # first=tasks[0][0]
        # rep.append(first)
        #print(rep)
        #print(len(tasks))
    new_tasks=[]
        # new_tasks.append(tasks[0])
    for i in range(len(tasks)):
        if tasks[i][0] not in rep:
            rep.append(tasks[i][0])
            new_tasks.append(tasks[i])
    return new_tasks

# 参数设置-----------------------------------------------------------------------
#gen = 1000  # 迭代次数
pc = 0.3  # 交叉概率
pm = 0.05  # 变异概率
popsize = 30  # 种群大小
n = 50  # 任务数量,即染色体长度n
# c = [5, 7, 9, 4, 3, 5, 6, 4, 7, 1, 8, 6, 1, 7, 2, 9, 5, 3, 2, 6]  # 每个物品的价值列表
#c = producetasks()#返回一个任务列表，有五十个任务的利润1-10
tasks_first=make_tasks()
# c=[]
# for i in range(len(tasks_first)):
#     c.append(tasks_first[i][3])
# print(c)
# print(len(c))
# if len(c==49):
#     c.append(random.randint(1,10))
##看到这里了
w = 5  # 每个物品所占据的重量
W = 175  # 存储空间
ttime = 6  # 每个任务花费的时间
# T = 3600 # 总时间区间

each_total_time = 1000  # 定义总时间
each_max_storage = 175 # 定义每个卫星的最大存储空间
fun = 1  # 1-第一种解码方式，2-第二种解码方式（惩罚项）

# 初始化-------------------------------------------------------------------------
# 初始化种群（编码）
##population是一个数组，每一个数组里是种群里个体
# 这个函数的目的是返回一个由‘0’，‘1’组成的二维数组
#{'0100001010...','10100010...',...}一共有三十个字符串，每个字符串里有50个由0，1组成的数字
population = init(popsize, n)
#print(f'ori{population}')
# 适应度评价（解码）
#value里一共有三十个元素，每个元素是一组50个0，1得出的利润
#ss是个二维数组，一共有三十个小组，每个小组里面是接受的每组任务的下标
if fun == 1:
    value,s = fitnessfun1(population, n, W, ttime, each_total_time)
    # print(value)
    # print(len(value))
    # print(s)
    # print(len(s))
    # print(f'zhong{zhong},{len(zhong)}')
    # print(f'profit{profit},{len(profit)}')
    # print(f'arr{arr},{len(arr)}')
    # print(f'end{end},{len(end)}')
# 初始化交叉个数
ncross = 0
# 初始化变异个数
nmut = 0
# 储存每代种群的最优值及其对应的个体
t = []
best_ind = []
last = []  # 储存最后一代个体的适应度值
realvalue = []  # 储存最后一代解码后的值

tot=100
# 循环---------------------------------------------------------------------------
t1 = time.time()
for i in range(tot):
    print("迭代次数：")
    print(i)
    # 交叉
    #print('交叉')
    offspring_c = crossover(population, pc, ncross)
  #  t2=time.time()-t1
  #  print(f'offspring_c{t2}')
    #print(offspring_c)
    # print(len(offspring_c))
    # 变异
    # offspring_m=mutation1(offspring,pm,nmut)
    #print('变异')
    offspring_m = mutation2(offspring_c, pm, nmut)
  #  t3 = time.time() - t2
   # print(f'offspring_m{t3}')
    # print()
    #print(offspring_m)
    # print(len(offspring_m))
    #print('混合')
    mixpopulation = population + offspring_m#一个60个组，每个组里有50个0，1组成的字符串
    # print(mixpopulation)
    # print(len(mixpopulation))
    # 适应度函数计算
    # value里一共有六十个元素，每个元素是一组50个0，1得出的利润
    # s是个二维数组，一共有六十个小组，每个小组里面是接受的每组任务的下标
    if fun == 1:
        value, s = fitnessfun1(mixpopulation, n, W, ttime, each_total_time)
       # t4 = time.time() - t3
      #  print(f'fitnessfun1{t4}')
    # print(value)
    # print(len(value))
    # print(s)
    # print(len(s))
    # print()
    # 轮盘赌选择########没看懂
    population = roulettewheel(mixpopulation, value, popsize)
   # t5 = time.time() - t4
   # print(f'roulett{t5}')
    # print(population)
    # print(len(population))
    # print()
    # 储存当代的最优解
    result = []
    if i == tot - 1:
        if fun == 1:
            value1, s1 = fitnessfun1(population, n, W, ttime, each_total_time)
            # t6 = time.time() - t5
            # print(f'offspring_c{t6}')
            realvalue = s1
            result = value1
            last = value1
    else:
        if fun == 1:
            value1, s1 = fitnessfun1(population, n, W, ttime, each_total_time)
            result = value1
    print(result)
    maxre = max(result)
    h = result.index(max(result))
    count=len(s1[h])
    count_list.append(count)
    avg_pro=maxre/count
    avg_each_tasks.append(avg_pro)
    #print(f'第{i}次最大的利润的下标是{h}')
    # 将每代的最优解加入结果种群
    t.append(maxre)  # 循环1000次
    #print(f'第{i}次最大的利润是{maxre}')
    total_list.append(maxre)
    summ += maxre
    avg = summ / (i + 1)
    avg_list.append(avg)


    best_ind.append(population[h])

# 输出结果-----------------------------------------------------------------------
if fun == 1:
    best_value = max(t)
    hh = t.index(max(t))#第几次循环最大的那个
   # f2, s2 = decode1(best_ind[hh], n, W, ttime, each_total_time)
    #f2为最好的一组的利润，s2为这一组接受任务的下标
    #print("此次最优组合为：")
    #print(s2)



    print("此次最优解为：")
    print(max(t))
    print("此次最优解出现的代数：")
    print(hh)
    t2 = time.time() - t1
    print("genetic algorithm consumed time %f" % t2)
    plot_profit(total_list, avg_list)
    plot_tasks(count_list)
    plot_avgtasks(avg_each_tasks)

   # t6=time.time()-t5

  #  print(f'fin{t6}')
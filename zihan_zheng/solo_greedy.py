import math
import random
import time
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib import pyplot as plt

from tasks import Tasks

max_stor=175  # the max storage of the satellite is 175
T = 450
count = 0
total_profit = 0
avg_list = []
total_list = []
count_list = []
avgprofit_list = []
num_list = []

def plot_avgtasks(avg):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Test episdoes")
    host.set_ylabel("Average profit")

    plt.title('greedy algorithm')

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
    host.set_xlabel("Testing episdoes")
    host.set_ylabel("Total reward")

    plt.title('greedy algorithm')

    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit", color = '#90EE90')
    p2, = host.plot(range(len(avg)), avg, label="Average profit", color = '#87CEEB')
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0, 225])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()


def plot_tasks(tasks):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")

    plt.title('50 tasks in 450s')
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

t1 = time.time()
for p in range(100):
    tasks=[]
    target = Tasks()
    tasks_one = target.produce_tasks()
    current_stor = 0
    accepted_tasks = []
    pick_tasks = random.sample(range(1, 151), 50)# 在150个任务中跳出50个
    #print(pick_tasks)
#print(pick_tasks)
#print(len(pick_tasks))

    for i in range(len(tasks_one)):
        if tasks_one[i][0] in pick_tasks:
            tasks.append(tasks_one[i])

            #print(tasks_one[i])
    # pick_tasks = random.sample(range(1, 151), 50)
    # tasks = []
    # for i in range(len(tasks_one)):
    #     if tasks_one[i][0] in pick_tasks:
    #         tasks.append(tasks_one[i])
    # tasks = sorted(tasks, key=(lambda x: x[6]))
    # rep = []
    # # first=tasks[0][0]
    # # rep.append(first)
    # # print(rep)
    # # print(len(tasks))
    # new_tasks = []
    # # new_tasks.append(tasks[0])
    # for i in range(len(tasks)):
    #     if tasks[i][0] not in rep:
    #         rep.append(tasks[i][0])
    #         new_tasks.append(tasks[i])

    for m in range(len(tasks)):
     # calculate the greedy number, the higher selection priority
        execu_percent = float(tasks[m][8] / T)
        stor_percent = float(tasks[m][4] / max_stor)
        a = float(execu_percent / stor_percent)
        greedy_num = float(tasks[m][3] / (execu_percent + stor_percent))
        tasks[m].append(float(greedy_num))
        #print(tasks[m])

    tasks = sorted(tasks, key=(lambda x: x[9]), reverse=True)
    #for i in range(len(tasks)):
        #print(tasks[i][0])
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

            #print(rep)
           # print(len(rep))
            #print(tasks[i+1])
            #tasks.pop(i+1)

    #print(new_tasks)
    #print(len(new_tasks))



    num = 0
    current_profit = 0

    current_stor = 0
    sche_time = []
    jishu_list=[]
    for m in range(T):
        sche_time.append(0)
    for m in range(50):
        conflitt = 0
        start_time = int(new_tasks[m][6])
        end_time = math.ceil(new_tasks[m][7])

        if end_time > T - 1:
            continue
        else:
            for n in range(int(start_time), math.ceil(end_time)):
                if sche_time[n] == 1:
                    conflitt = 1
                    break
        if conflitt == 1:
            continue
        #
        if current_stor + new_tasks[m][4] > max_stor:
            continue
        #
        jishu_list.append(m)
        count += 1  ##记录接受了几个任务
        current_profit += new_tasks[m][3]
        current_stor += new_tasks[m][4]
        num += 1
        for n in range(int(start_time), math.ceil(end_time)):
            sche_time[n] = 1
    print(f'jishu_list{jishu_list}')
    print(f'接受了{num}个任务，目前的总利润是{current_profit}')
    #print(f'接受了{count}个任务，目前的总利润是{current_profit}')
    # print("%d,   %d" % (count, current_profit))
    total_profit += current_profit
    avg_profit_h = current_profit /num
    avgprofit_list.append(avg_profit_h)
    #print(f'avg{avg_profit_h}')
    avg = total_profit / (p + 1)
    #print(total_profit)

    count_list.append(count)
    total_list.append(current_profit)
    #avg_list.append(current_profit)
    avg_list.append(avg)
    num_list.append(num)
    count = 0
    #print(avg_list)

t2 = (time.time()-t1)
print(f'greedy algorithm consumed time ： {t2}')

plot_profit(total_list, avg_list)#平均利润
plot_tasks(count_list)#每次接受的任务数
plot_avgtasks(avgprofit_list)#每个任务的平均利润
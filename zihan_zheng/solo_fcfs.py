import random
import time
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib import pyplot as plt

from tasks import Tasks

avg_profit_fcfs=[]
num_list_fcfs=[]
total_list_fcfs=[]
avg_list_fcfs=[]
current_profit_list=[]
max_stor=175  # the max storage of the satellite is 175
T = 450
total_profit_fcfs = 0
toatl_num_fcfs = 0

def plot_avgtasks(avg):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Test episdoes")
    host.set_ylabel("Average profit")

    plt.title('fcfs algorithm')

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

    plt.title('fcfs algorithm')

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
for q in range(100):
    current_profit = 0
    current_stor = 0
    toatl_num_fcfs = 0
    target = Tasks()
    tasks_one = target.produce_tasks()
    pick_tasks = random.sample(range(1, 151), 50)
    tasks = []
    for i in range(len(tasks_one)):
        if tasks_one[i][0] in pick_tasks:
            tasks.append(tasks_one[i])
    tasks = sorted(tasks, key=(lambda x: x[6]))
    rep = []
    # first=tasks[0][0]
    # rep.append(first)
    # print(rep)
    # print(len(tasks))
    new_tasks = []
    # new_tasks.append(tasks[0])
    for i in range(len(tasks)):
        if tasks[i][0] not in rep:
            rep.append(tasks[i][0])
            new_tasks.append(tasks[i])

    accepted_tasks=[]
    #tasks = [[] for j in range(50)]  # this means that the total number of tasks that should be scheduled is 50
    # each task's format:  ['arrival time', 'execution time', 'storage', 'profit']
    #arr_time = []
    #

    # for l in range(50):
    #     temp_execu = random.randint(10, 30)
    #     temp_profit = random.randint(1, 10)
    #     tasks[l].append(arr_time[l])
    #     tasks[l].append(temp_execu)
    #     tasks[l].append(stor_fcfs)
    #     tasks[l].append(temp_profit)

    # above part is to create the tasks that need to be scheduled
    # 第四列为在任务收益;
    # 第五列为在任务所需内存;
    # 第六列为在该卫星中经过目标点的轮数;
    # 第七列为卫星开始观测的时间，
    # 第八列是结束观测的时间，
    jishu_fcfs=[]
    current_time = 0 # this time is used to give the specific time that will not cause any conflicts
    for m in range(50):
        if m == 0 and new_tasks[m][7]<T:
            accepted_tasks.append(m)
            current_stor+=new_tasks[m][4]
            current_profit+=new_tasks[m][3]
            current_time = new_tasks[m][7]
            toatl_num_fcfs += 1
            jishu_fcfs.append(m)
        else:
            if (current_stor + new_tasks[m][4])<max_stor and current_time<new_tasks[m][6] and new_tasks[m][7]<T :
                jishu_fcfs.append(m)
                accepted_tasks.append(m)
                current_stor += new_tasks[m][4]
                current_profit += new_tasks[m][3]
                current_time = new_tasks[m][7]
                toatl_num_fcfs+=1
            else:
                continue
    #jishu_fcfs.append(m)
    print(f'jishu_fcfs{jishu_fcfs}')
    avgprofit_fcfx = current_profit/toatl_num_fcfs #每个任务的平均利润
    avg_profit_fcfs.append(avgprofit_fcfx)   #每个任务的平均利润列表
    #print(f'avg{avgprofit_fcfx}')
    num_list_fcfs.append(toatl_num_fcfs) #接受任务数
    current_profit_list.append(current_profit)
    #print(current_profit)
    total_profit_fcfs+=current_profit  #总利润
    #print(total_profit_fcfs)

    total_list_fcfs.append(current_profit) #每次总利润列表
    #avg = current_profit / (i + 1)
    avg = total_profit_fcfs/(q+1) #平均利润
    #print(f'avg {avg}')
    avg_list_fcfs.append(avg)#平均利润列表
   # avg_list_fcfs.append(current_profit)
    print(f'fcfs current profit {current_profit},fcfs accepted tasks{toatl_num_fcfs}')
    #print(f'fcfs accepted tasks{toatl_num_fcfs}')
t2 = (time.time() - t1)
print(f'fcfs consumed time ： {t2}')

plot_profit(current_profit_list, avg_list_fcfs)#平均利润
plot_tasks(num_list_fcfs)#每次接受的任务数
plot_avgtasks(avg_profit_fcfs)#每个任务的平均利润
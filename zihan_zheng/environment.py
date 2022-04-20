import numpy as np
import random
import copy
from tasks import Tasks
from collections import Counter

class Environment(object):
# INTRODUCING AND INITIALIZING ALL THE PARAMETERS AND VARIABLES OF THE ENVIRONMENT
    def __init__(self):  # 定义所需的变量
        """"""
        """定义任务"""
        self.target = Tasks() #实体化Tasks()类
        self.tasks_one = self.target.produce_tasks() #调用target.produce_tasks函数，生成任务列表
        self.task_accepted = [] #用于记录已经接受的任务
        self.tasks = [] #在150个任务中随机挑出50个任务用于训练
        """定义卫星"""
        self.each_total_time = 1000 #定义总时间
        self.each_max_storage = 175 #定义每个卫星的最大存储空间
        self.satellites = [0 for i in range(3)] #【开始时间。结束时间，剩余空间】
        self.transfer_time = 0.5
        self.observe_time = 4


    def reset(self): #初始化卫星参数和50个任务
        # 初始化卫星
        self.satellites = [0,self.each_total_time,self.each_max_storage]
        self.f_satellites = [0, self.each_total_time, self.each_max_storage]
        #初始化任务
        self.tasks = []
        self.fcfs_task = []#初始化fcfs的接受的任务的列表
        self.task_accepted =[]
        #print("len(tasks_one)")
        #print(len(self.tasks_one))
        #随机挑选出50个任务
        pick_tasks = random.sample(range(0, 151), 50)#在150个任务中跳出50个
        for i in range(len(self.tasks_one)):
            if self.tasks_one[i][0] in pick_tasks:
                self.tasks.append(self.tasks_one[i])
        # 对任务按开始观测时间进行排序
        #task = np.array(self.tasks) #转换tasks的类型，方便排序
        #index = np.lexsort([task[:, 5]]) #找到第6列应该排序的下标
        #self.tasks = task[index, :]
        self.row_tasks = len(self.tasks)#/////////这个变量是什么意思啊？
        self.state=[0 for i in range(6)]
        self.next_state = [0 for i in range(6)]
        self.profit_total = 0

        #self.task_accepted.append(self.tasks[0][0])


    def observe(self):
        self.counts =0   #/////////////self.counts什么意思
        self.state[0] = (self.tasks[self.counts][1] + 60)/ 120  # 纬度
        self.state[1] = (self.tasks[self.counts][2] + 30) / 60  # 经度
        self.state[2] = self.tasks[self.counts][3] / 10  # 单步收益   ###单步收益，单占用内存和剩余时间窗口数是什么意思
        self.state[3] = self.tasks[self.counts][4] / 10  # 单占用内存
        self.state[4] = self.tasks[self.counts][5] / 10  # 剩余时间窗口数量
        # 剩余时间窗口的数量#///////////////////底下的代码怎么理解？？？？？？？？？？？
        targets = [tmp[0] for tmp in self.tasks[self.counts:]]
        count = Counter(targets)
        targets_wnum = count[self.tasks[self.counts][0]]
        #self.state[3] = targets_wnum / 10
        self.state[5] = 0  # 是否接受
        return self.state


#判断任务是否可以被卫星执行
    def is_accept(self,task_lable): #task_lable是任务标签
        self.profit = 0  # 存放n_agents个的收益
        self.counts = task_lable
        accept = 0  # 用于存储是否有资源接受任务，有为0，无则为1
        #if self.tasks[self.counts][0] in self.task_accepted:
            #accept = 1
        # 分配起止时间 满足论文75页 唯一性观测需求
        if self.satellites[2] < self.tasks[self.counts][4]: #空闲内存不足
            accept = 1
        else: #判断时间窗口 满足论文75页 连续观测任务约束//？？？？？？？连续观测任务是什么？
            q = self.tasks[self.counts][6]  # 当前任务开始时间
            t = 2 * self.transfer_time + self.observe_time  #当前任务结束时间？？？？？？不需要加上任务开始时间吗？
            if self.satellites[0] > q or self.satellites[1] < t: #任务的开始时间/结束时间不对
                #///////////////这个卫星的开始时间和结束时间有变过吗？还是开始时间一直是0，结束时间也是固定的
                #要在卫星的时间窗口内执行，在卫星开始观测之后才能观测，在卫星观测结束之前结束观测
                accept = 1
        return accept


############################
# 更新环境
#///////////这个函数看不懂，self.counts是什么
    def update_env(self, action_n,accept):
        self.done = False
        count = self.counts + 1
        if (action_n == 0 and accept == 0):
            self.task_accepted.append(self.tasks[self.counts][0])
            self.next_state[5] = 0
            self.profit = self.tasks[self.counts][3]
            self.done = True
            self.satellites[0] = self.tasks[self.counts][6] + self.observe_time
            self.satellites[2] = self.satellites[2] - self.tasks[self.counts][4]
        elif (action_n == 0 and accept == 1):
            self.next_state[5] = 1
            self.profit = 0
            self.done = False
        else:
            self.next_state[5] = 1
            self.profit = 0
            self.done = False
        self.next_state[0] = (self.tasks[count][1] + 60) / 120
        self.next_state[1] = (self.tasks[count][2] + 30) / 60
        self.next_state[2] = self.tasks[count][3] / 10
        self.next_state[3] = self.tasks[count][4] / 8
        self.next_state[4] = self.tasks[count][5] / 10
        return np.array(self.next_state), self.profit, self.done



#用来记录fcfs是否接受任务
    def f_is_accept(self,task_lable):
        self.counts = task_lable
        accept = 0  # 用于存储是否有资源接受任务，有为0，无则为1
        if self.tasks[self.counts][0] in self.fcfs_task:
            accept = 1
        # 分配起止时间 满足论文75页 唯一性观测需求
        if self.f_satellites[2] < self.tasks[self.counts][4]: #空闲内存不足
            accept = 1
        else: #判断时间窗口 满足论文75页 连续观测任务约束
            q = self.tasks[self.counts][6]  # 当前任务开始时间
            t = 2 * self.transfer_time + self.observe_time  #当前任务结束时间
            if self.f_satellites[0] > q or self.satellites[1] < t: #任务的开始时间/结束时间不对
                accept = 1
        return accept
    def fcfs(self,is_accept):
        if is_accept == 0:
            self.f_satellites[0] = self.tasks[self.counts][6] + self.observe_time
            self.f_satellites[2] = self.f_satellites[2] - self.tasks[self.counts][4]
            self.fcfs_task.append(self.tasks[self.counts][0])
            profit = self.tasks[self.counts][3]
        else:
            profit = 0
        return profit







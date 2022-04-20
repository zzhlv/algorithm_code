import copy

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from environment import Environment
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import time
#from torch.utils.tensorboard import SummaryWriter
import os
from collections import Counter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#定义超参数
MAX_EPISODES = 100
MAX_TASKS = 50#任务数量
LR_A = 0.001#actor网络的学习率
LR_C = 0.002#critic网络的学习率
GAMMA = 0.9#奖励折扣
TAU = 0.3#软替换值
MEMORY_CAPACITY = 10000#内存容量
BATCH_SIZE = 32#批量大小

RENEDR=False

#设置各个列表
eval_profit_list=[]
avg_profit_list=[]
eval_tasks_list=[]
avg_tasks_list=[]
eval_ap_list=[]
avg_ap_list=[]

class Actor(torch.nn.Module):#定义actor网络
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(s_dim, 128)#全连接层，进行线性变换，升维
        self.fc1.weight.data.normal_(0, 0.01)#初始化权重参数
        self.fc2 = torch.nn.Linear(128, a_dim)#降维
        self.fc1.weight.data.normal_(0, 0.01)

    def forward(self,state_input):
        self.net=F.relu(self.fc1(state_input))#激活函数,修正线性单元
        self.a=F.relu(self.fc2(self.net))#激活函数
        return self.a

class Critic(torch.nn.Module):#定义critic网络
    def __init__(self):
        super(Critic, self).__init__()
        self.w1_s = nn.Linear(s_dim, 128)
        self.w1_s.weight.data.normal_(0, 0.01)
        self.w1_a = nn.Linear(a_dim, 128)
        self.w1_a.weight.data.normal_(0, 0.01)
        self.out = nn.Linear(128, 1)
        self.out.weight.data.normal_(0, 0.01)

    def forward(self,s,a):
        net=F.relu(self.w1_s(s)+self.w1_a(a))
        return self.out(net)



class DDPG(object):#定义ddpg网络结构
    def __init__(self,a_dim,s_dim):
        self.memory=torch.zeros((MEMORY_CAPACITY,s_dim*2+a_dim+1),dtype=torch.float32)#初始化memory，返回一个memory——capacity行，10列的，里面的每一个值都是0的tensor
        self.pointer=0
        self.a_dim,self.s_dim=a_dim,s_dim

        #initialize actor net
        self.actor_eval = Actor()
        self.actor_target = Actor()

        #initialize crititc net
        self.critic_eval = Critic()
        self.critic_target = Critic()

        #Updating parameters of random gradient descent algorithm
        self.ae_optimizer = torch.optim.Adam(params=self.actor_eval.parameters(), lr=0.001)
        self.ce_optimizer = torch.optim.Adam(params=self.critic_eval.parameters(), lr=0.001)
        #calculate error
        self.mse = nn.MSELoss()

    def return_c_loss(self,S,a,R,S_):#计算caitic网络的误差
        a_=self.actor_target(S_)#下一步的action
        q = self.critic_eval(S, a)#实际的奖励
        q_ = self.critic_target(S_, a_)#预测的奖励
        q_target = R + GAMMA * q_
        td_error = self.mse(q_target, q)#计算均方误差Calculate mean square error
        return td_error

    def return_a_loss(self,S):#计算actor网络的误差
        #loss_a = -torch.mean(q)
        a=self.actor_eval(S)#在状态S下的action
        q=self.critic_eval(S,a)#caitic网络对当前动作的评价
        a_loss= -q.mean()#通过求平均数的方式来计算误差
        return a_loss

    def choose_action(self,s):#选择动作
        a = self.actor_eval(s[np.newaxis,:])[0]
        a = np.clip(np.random.normal(a.detach().numpy(), var), -1, 1)  # 将tensor动作转换为0或1
        a = int((a + 2) / 4 + 0.5)
        # print("!",s)
        # a=np.clip(np.random.normal(a.detach().numpy(),var),-1,1)#将tensor动作转换为0或1
        # a=a.detach()#取消a的梯度，使其不具有梯度属性
        # a=a.numpy()#将a转换成numpy数组形式，提高运算速度
        # a=np.random.normal(a,var)#正态分布，a为正态分布的均值，var为正态分布的标准差
        # a=np.clip(a,-1,1)#将a限制在（-1,1）之间，小于-1就为-1，大于1的就为1
        # a=int((a+2)/4+0.5)#为动作添加随机性
        return a

    def learn(self):#学习过程
        convert_eval_to_target(self.actor_eval,self.actor_target)#将actor网络的eval的参数传递给target
        convert_eval_to_target(self.critic_eval,self.critic_target)#将caitic网络的eval的参数传递给target
        indices=np.random.choice(MEMORY_CAPACITY,size=BATCH_SIZE)#从memory capacity中选择一批数据进行训练
        bt=self.memory[indices,:]
        bs = bt[:, :self.s_dim]#将状态分割出来
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]#将动作分割出来
        br = bt[:, -self.s_dim - 1: -self.s_dim]#将奖励分割出来
        bs_ = bt[:, -self.s_dim:]#将下一个状态分割出来

        a_loss = self.return_a_loss(bs)  # 计算actor网络的误差
        loss_save_a.append(a_loss.detach().numpy())



        #actor网络优化器
        self.ae_optimizer.zero_grad()#将梯度置0
        a_loss.backward()#利用反向传播算法更新参数Updating parameters using back propagation algorithm
        self.ae_optimizer.step()

        c_loss = self.return_c_loss(bs, ba, br, bs_)  # 计算caitic网络的误差
        #critic网络优化器
        self.ce_optimizer.zero_grad()#将梯度置0
        c_loss.backward()#利用反向传播算法更新参数
        loss_save_q.append(c_loss.detach().numpy())
        self.ce_optimizer.step()




        # print(q)
        # print(loss_a)

    def store_transition(self,s,a,r,s_):
        transition=torch.FloatTensor(np.hstack((s,a,[r],s_)))
        index=self.pointer%MEMORY_CAPACITY#使用新的memory替换之前的memory
        self.memory[index,:]=transition
        self.pointer+=1

def plot_profit(profit,avg):#将总奖励画图
    host=host_subplot(111)#设置第一张图片的row=1，col=1
    plt.subplots_adjust(right=0.8)#调整绘图窗口的右边界

    #设置标签
    host.set_xlabel("Training Episdoes")#X轴
    host.set_ylabel("Total Profit")#Y轴

    #绘制曲线
    p1,=host.plot(range(len(profit)),profit,label="Total Profit")
    p2,=host.plot(range(len(avg)),avg,label="Running Average Total Profit")

    #设置图例的位置
    host.legend(loc=1)

    #设置标签的颜色
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())

    #设置x轴和y轴的范围
    host.set_xlim([0, MAX_EPISODES])
    host.set_ylim([0, 100])

    plt.draw()
    plt.show()

def plot_tasks(tasks,avg1):#将接受的任务总数画图
    host = host_subplot(111)  # 设置第一张图片的row=1，col=1
    plt.subplots_adjust(right=0.8)  # 调整绘图窗口的右边界

    #设置标签
    host.set_xlabel("Training Episdoes")
    host.set_ylabel("Total Accepted Tasks")

    #绘制曲线
    p1,=host.plot(range(len(tasks)),tasks,label="Total Accepted Tasks")
    p2,=host.plot(range(len(avg1)),avg1,label="Running Average Total Accepted Tasks")

    # 设置图例的位置
    host.legend(loc=1)

    #设置标签的颜色
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())

    # 设置x轴和y轴的范围
    host.set_xlim([0, MAX_EPISODES])
    host.set_ylim([0, 30])

    plt.draw()
    plt.show()

def plot_ap(ap,avg2):#将平均利润画图
    host = host_subplot(111)  # 设置第一张图片的row=1，col=1
    plt.subplots_adjust(right=0.8)  # 调整绘图窗口的右边界

    # 设置标签
    host.set_xlabel("Training Episdoes")
    host.set_ylabel("Average Profit")

    #绘制曲线
    p1,=host.plot(range(len(ap)),ap,label="Average Profit")
    p2,=host.plot(range(len(avg2)),avg2,label="Running Average Profit")

    # 设置图例的位置
    host.legend(loc=1)

    # 设置标签的颜色
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())

    # 设置x轴和y轴的范围
    host.set_xlim([0, MAX_EPISODES])
    host.set_ylim([0, 9])

    plt.draw()
    plt.show()

def convert_eval_to_target(e,t):#将eval的参数传递给target
    for x in t.state_dict().keys():
        eval('t.'+x+'.data.mul_((1-TAU))')
        eval('t.'+x+'.data.add_(TAU*e.'+x+'.data)')

#############################  Training  ###########################
t1_time=time.time()
if __name__ == '__main__':

    env=Environment()#首先初始化环境，创建100个任务

    a_dim = 1
    s_dim=6

    ddpg=DDPG(a_dim,s_dim)#初始化神经网络，包括memory，actor（eval target）和critic（eval target）
    #print(a_dim)
    #print(s_dim)
    loss_save_a = []#用于存储loss_a
    loss_save_q = []  # 用于存储loss_q


    # writer=SummaryWriter("./logs_ddpg")
    # writer.add_graph(ddpg,1,4)
    # writer.close()

    var=3
    R=0
    T=0

    for episode in range(MAX_EPISODES):#最外层一共2000次循环
        s=env.reset()#重置初始环境状态
        start=time.time()#获取开始时间
        env.atasks = 0
        total_profit = 0
        total_f = 0
        done_num = []
        a_a = []
        #MAX_TASKS = env.row_tasks - 1
        #print('env',env.row_tasks-1)
        old_task=[]#用于记录经过的任务
        for j in range(MAX_TASKS):#内层100次循环
            s=env.observe()#加载环境
            a=ddpg.choose_action(torch.FloatTensor(s))#根据环境选择动作,此时输出的a是具有梯度的
            a_a.append(a) #将动作加入动作列表中
            #old_task.append(env.tasks[j][0]) #将任务加入任务列表中
           #old_task_count = Counter(old_task)
                ##print("重复啦")
            accept = env.is_accept(j)
            #env.atasks+=accept
            #print('et',env.atasks)
            f_accept= env.f_is_accept(j)
            #先来先服务
            fcfs_profit = env.fcfs(f_accept)
            total_f = total_f + fcfs_profit
            if env.tasks[j][0] not in env.task_accepted:#已经接受的任务不再参与和后面的训练
                #r_time = old_task_count[env.tasks[j][0]] * 1
                s_, r, done = env.update_env(a, accept)  # 根据action获取s_，r，done
                total_profit += r
                done_num.append(done)
                ddpg.store_transition(s, a, r / 10, s_)
                s = s_
                ##print("重复啦")


            if j== MAX_TASKS-1:#外层每循环一次，输出一次训练结果
                print('Episode:',episode,'Reward_d:',total_profit,'Accepted Tasks_d:',len(env.task_accepted))
                print('Episode:', episode, 'Reward_f:', total_f, 'Accepted Tasks_f:', len(env.fcfs_task))
                print(env.task_accepted)
                print(a_a)
                # print(len(a_a))
                # print(MAX_TASKS)
                break

            if ddpg.pointer>MEMORY_CAPACITY:#外层每循环20次（memory中有2000组数据），选取其中32组数据对神经网络参数进行更新
                var *= 0.9995 #减弱动作的随机性
                ddpg.learn()

        end=time.time()
        times=end-start
        #print("外围的a_a:")
       # print(a_a)
        T=T+times
        #print('T',T)
        total_reward = total_profit
        accepted_tasks = len(env.task_accepted)
        eval_profit_list.append(total_reward)
        avg = sum(eval_profit_list) / len(eval_profit_list) #平均收益
        avg_profit_list.append(avg)
        eval_tasks_list.append(accepted_tasks)
        avg_1 = sum(eval_tasks_list) / len(eval_tasks_list) #Average accepted tasks
        avg_tasks_list.append(avg_1)
        avg_profit = total_reward /( accepted_tasks+1)
        eval_ap_list.append(avg_profit)
        avg_2 = sum(eval_ap_list) / len(eval_ap_list) #Average Profit
        avg_ap_list.append(avg_2)
    t2_time = time.time()
    print('ddpg consumed time:',t2_time-t1_time)
    print('Average Total Profit', avg)
    print('Average accepted tasks', avg_1)
    print('Average Profit', avg_2)
    print('Average Time', T / MAX_EPISODES)

    plot_profit(eval_profit_list, avg_profit_list)
    plot_tasks(eval_tasks_list, avg_tasks_list)
    plot_ap(eval_ap_list, avg_ap_list)
    np.savetxt("a.txt", loss_save_a)
    np.savetxt("c.txt", loss_save_q)

import xlrd
import numpy as np
import copy
from collections import Counter


##########################将任务从excel中导出，并将任务生成一个二维数组的形式， ################################
#该文件最后生成的是target_update二维数组，作为produce_tasks的返回值返回
#其中，每一行表示一个卫星对目标的观测数据;
# 第一列是任务下标;
# 第二列为纬度  针对任务来说
# 第三列为经度
# 第四列为在任务收益;
# 第五列为在任务所需内存;
# 第六列为在该卫星中经过目标点的轮数;
# 第七列为卫星开始观测的时间，
# 第八列是结束观测的时间，
# 第九列为持续时间
#


# 该任务可以观测的时间
# 1-5指的是1个任务
# 任务全天，看卫星有没有时间

class Tasks(object):
    def __init__(self):
        self.target = []
    def produce_tasks(self):
        data = xlrd.open_workbook("easy.xls")  # 打开当前目录下名为easy.xlsx的文档
        # 此时data相当于指向该文件的指针
        table = data.sheet_by_index(0)  # 通过索引获取，例如打开第一个sheet表格
        table = data.sheet_by_name("1")  # 通过名称获取，如读取sheet1表单
        table = data.sheets()[0]  # 通过索引顺序获取
        # 以上三个函数都会返回一个xlrd.sheet.Sheet()对象
        names = data.sheet_names()  # 返回book中所有工作表的名字
        #任务下标
        target_list = [[1],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[2],[20],[21],[22],[23],[24],[25],[26],[27],[28],[29],
                       [3],[30],[31],[32],[33],[34],[35],[36],[37],[38],[39],[4],[40],[41],[42],[43],[44],[45],[46],[47],[48],[49],
                       [5],[50],[6],[7],[8],[9],[51],[52],[53],[54],[55],[56],[57],[58],[59],[60],[61],[62],[63],[64],[65],[66],
                       [67],[68],[69],[70],[71],[72],[73],[74],[75],[76],
                       [77],[78],[79],[80],[81],[82],[83],[84],[85],[86],[87],[88],[89],[90],[91],[92],[93],[94],[95],[96],[97],[98],
                       [99],[100],[101],[102],[103],[104],[105],[106],[107],[108],[109],[110],[111],[112],[113],[114],[115],[116],
                       [117],[118],[119],[120],[121],[122],[123],[124],[125],[126],[127],[128],[129],[130],[131],[132],[133],[134],
                       [135],[136],[137],[138],[139],[140],[141],[142],[143],[144],[145],[146],[147],[148],[149],[150]]#二维数组
        geographic = xlrd.open_workbook("geographic.xls") # 打开当前目录下名为geographic.xls的文档
        geo_table = geographic.sheets()[0]  # 通过索引顺序获取
        geo_nrows = geo_table.nrows  # 获取该sheet中的有效行数
        #更加target_list的前3列
        #就是把数据表中的数据插入target_list
        for g in range(geo_nrows):  #将序号和地理位置相对应，建立一个4
            temp =[]
            geo_temp = geo_table.row_values(g)
            temp.append(int(geo_temp[0]))
            geo_index_target_list = target_list.index(temp)
            target_list[geo_index_target_list].append(geo_temp[1])
            target_list[geo_index_target_list].append(geo_temp[2])

        tol_store = 0
######################给任务列表增加收益 第四列，第五列
        for i in range(len(target_list)):
            #给任务增加收益
            profit = np.random.randint(1,10)
            target_store = np.random.randint(1, 8)
            tol_store+=target_store
            target_list[i].append(profit)
            target_list[i].append(target_store)
        #print(target_list)



#####################将卫星观测数据增加到任务列表中
        nrows = table.nrows  #获取该sheet中的有效行数
        target_lable = -0.5 #记录目标的下标
        target_temp = [0 for i in range(len(target_list[0]))]#[0,0,0,0,0,0,0,0,0,0]
        for j in range(nrows-1):
            a = table.row(j)  #返回由该行中所有的单元格对象组成的列表
            if a[1].ctype == 0 or a[1].ctype == 1: #除去excel中无效的行，如空行或者标题行
                target_lable += 0.5
                continue
            target_lable = int(target_lable) #记录卫星个数
            try:
                target_temp = copy.deepcopy(target_list[target_lable])
                target_temp.extend(table.row_values(j)) #将卫星下标和卫星的相关信息结合在一起，形成一个数组
                self.target.append(target_temp)
#将卫星信息打包成一个二维数组，其中，每一行表示一个卫星;第一列是卫星下标;第二、三列为在任务位置，第四列为在任务收益;第五列为在该卫星中经过目标点的轮数?????难道不应该是单个存储量吗？;第六列为卫星开始观测的时间，第七列是结束观测的时间，第八列为持续时间
            except:
                print('第{0}条数据处理失败'.format(j))
        #print(self.target)
        #将第6列添加到列表中
        ##第六列是干什么的？
        for j in range(len(self.target) - 1):
            targets_list_label = [tmp[0] for tmp in self.target[j:]]
            count = Counter(targets_list_label)
            a  = count[self.target[j][0]]
            self.target[j].append(a)
        #print(type(a))
        #print(self.target)
#[[1, 30.0, -10.0, 2, 1, 1.0, 44520.28377226852, 44520.29272359954, 773.395, 5], [1, 30.0, -10.0, 2, 1, 2.0, 44520.35233262731, 44520.36287863426, 911.176, 4], [1, 30.0, -10.0, 2, 1, 3.0, 44520.799399282405, 44520.80808423611, 750.38, 3], [1, 30.0, -10.0, 2, 1, 4.0, 44520.86827827546, 44520.87884157407, 912.669, 2], [1, 30.0, -10.0, 2, 1, 5.0, 44520.94014769676, 44520.94465491898, 389.424, 1],
        # 第一列是卫星下标;
        # 第二、三列为在任务位置
        # 第四列为在任务收益
        # 第五列为单个存储量
        # 第六列为该卫星中经过目标点的轮数
        # 第七列为卫星开始观测的时间，第八列是结束观测的时间，第九列为持续时间
        # 第十列为？？？？？

##########################将上面生成的任务进一步进行处理 ################################
        start_time = 44520 #2021/11/20 0:00时间转换为数字
        compression_index = 100
        target_update = [[0 for i in range(len(self.target[0])-1)] for j in range(len(self.target )-1)]
        #print(target_update)

        for i in range(len(self.target)-1):
            try:
                target_update[i][0] = self.target[i][0]
                target_update[i][1] = self.target[i][1]
                target_update[i][2] = self.target[i][2]
                target_update[i][3] = self.target[i][3]
                target_update[i][4] = self.target[i][4]
                target_update[i][5] = self.target[i][9]
                target_update[i][6] = (self.target[i][6] - start_time) * 1200
                target_update[i][7] = (self.target[i][7] - start_time) * 1200
                target_update[i][8] = self.target[i][8] / 86400 * 1200

               #的 target_update[i][9] = self.target[i][3] /(self.target[i][4]/tol_store)
            except:
                print('第{0}条数据处理失败'.format(i))
        #print(target_update)
##########################将任务进行排序，并将第5列进行处理 ################################

        # target = np.array(target_update)  # 转换tasks的类型，方便排序
        # index = np.lexsort([target[:, 6]])  # 找到第6列应该排序的下标
        # target_update= target[index, :]




        return target_update






a = Tasks()

target1=a.produce_tasks()
#print(target1)
#

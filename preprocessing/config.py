import math
DATA_PATHA = "../data"
FLAG = "weibo"
DATA_PATHA = "../data/"+FLAG
max_NumNode = 128
n_time_interval = 64

if(FLAG == "weibo"):
    cascades = DATA_PATHA + "/dataset_weibo.txt"
    cascade_train = DATA_PATHA + "/cascade.txt"
    shortestpath_train = DATA_PATHA + "/shortestpath.txt"
    longpath = DATA_PATHA + "/longpath.txt"
    index_mapping_txt= DATA_PATHA + "/weibo_id.txt"

    # observation 表示一个观察时间窗口的长度，以秒为单位。 29分59秒
    # observation = 0.5 * 60 * 60 - 1
    observation = 0.5 * 60 * 60 - 1

    # time_interval 表示时间间隔的长度，用于将观察时间划分为多个等长的子时间段。 30分
    time_interval = math.ceil((observation + 1) * 1.0 / n_time_interval)  # 向上取整

    # pre_times 是一个列表，其中的每个元素表示一个时间点（以秒为单位），通常用于预测或计算在特定时间点之前发生的事件数量。
    # 一天
    pre_times = [24 * 3600]
    train_pkl = DATA_PATHA + "/data.pkl"

if(FLAG=="twitter"):

    cascades = DATA_PATHA + "/dataset.txt"
    cascade_train = DATA_PATHA + "/cascade.txt"
    shortestpath_train = DATA_PATHA + "/shortestpath.txt"

    observation = 3600*24*3
    time_interval = math.ceil((observation + 1) * 1.0 / n_time_interval)
    pre_times = [2764800]
    train_pkl = DATA_PATHA + "/data.pkl"

print("dataset:",FLAG)
print("observation time", observation)
print("the number of time slots:", n_time_interval)




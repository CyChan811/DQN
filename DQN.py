import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from collections import deque
import time
import math
from matplotlib import pyplot as plt
from matplotlib import ticker as tk

# 地图
maze = torch.tensor(
    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
     [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float
)

# maze = torch.tensor(
#     [[0., 0., 0., 0., 0.],
#      [0., 0., 1., 0., 0.],
#      [0., 1., 0., 1., 0.],
#      [0., 0., 0., 0., 0.],
#      [0., 0., 0., 0., 0.]]
# )

# 初始化
action_num = 4  # 4个动作，上、下、左、右
start_state_pos = (0, 0)   # 起点
target_state_pos = (10, 14)  # 终点
actions = (0, 1, 2, 3)   # 上下左右
POS_VALUE = 2   # 当走到(row,col)时，令迷宫矩阵在(row,col)处的值为POS_VALUE
batch_size = 200    # 批处理数目
tao = 0.3  # “软”更新目标Q网络权重的参数  ω-←τω+（1-τ）ω-
memory_size1 = 2000   # 正奖励序列记忆库大小
memory_size2 = 2000   # 负奖励或零奖励序列记忆库大小
rho = 0.3    # 以0.3概论从正奖励序列中抽取样本
test_flg = 0

episode_list = []
step_list = []

# 计算地图大小
pf_maze_size = math.sqrt(math.pow(maze.shape[0], 2) + math.pow(maze.shape[1], 2))
# 计算各个点与终点的距离
maze_distance = np.zeros((maze.shape[0], maze.shape[1]), dtype=np.int)
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        maze_distance[i][j] = math.sqrt(math.pow((j - target_state_pos[1]), 2) + math.pow((i - target_state_pos[0]), 2))
print(maze_distance)

# 将迷宫矩阵转为图片格式的shape(channel,height,width)
def matrix_to_img(row, col):
    state = copy.deepcopy(maze)
    state[row, col] = POS_VALUE
    # 维度转换
    state = np.reshape(state, newshape=(1, 1, state.shape[0], state.shape[1]))
    state = torch.FloatTensor(state)
    return state


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)    # 16个输出，卷积核是3*3
        self.row_count = maze.shape[0] - 3 + 1
        self.column_count = maze.shape[1] - 3 + 1
        self.conv2 = nn.Conv2d(16, 32, 3)   # 32个输出，卷积核是3*3
        self.row_count = self.row_count - 3 + 1
        self.column_count = self.column_count - 3 + 1
        self.fc1 = nn.Linear(32 * self.row_count * self.column_count, 64)     # 64个输出
        self.fc2 = nn.Linear(64, 4)     # 4个输出，对于4个动作的Q值

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * self.row_count * self.column_count)  # 把卷积核拉直，为了全连接
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()   # 创建当前网络和目标网络
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.epsilon = 1  # 动作探索值
        self.epsilon_min = 0.1  # epsilon最小值
        self.epsilon_decay = 0.998  # 探索衰减值
        self.gamma = 0.9  # 奖励衰减函数
        self.memory_count1 = 0
        self.memory1 = deque(maxlen=memory_size1)  # 正奖励序列记忆库
        self.memory_count2 = 0
        self.memory2 = deque(maxlen=memory_size2)  # 负奖励或零奖励序列记忆库
        self.optimizer = optim.SGD(self.eval_net.parameters(), lr=0.1)     # 随机梯度下降优化器
        self.loss_func = nn.MSELoss()   # 误差函数

    # 动作选择函数
    def choose_actions(self, x, current_row, current_col):
        if np.random.uniform() < self.epsilon:  #结合人工势场
            pf_f, pf_action = self.potential_field(current_row, current_col)
            pf_add_num = int(pf_f)
            choose_num = np.random.randint(0, 100 + pf_add_num)
            # choose_num = np.random.randint(0, 100)
            if 0 <= choose_num < 25:
                action = 0
            elif 25 <= choose_num < 50:
                action = 1
            elif 50 <= choose_num < 75:
                action = 2
            elif 75 <= choose_num < 100:
                action = 3
            else:
                action = pf_action
        else:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]  # 选出最大值
        return action

    # 人工势场
    def potential_field(self, y, x):
        pf_k = 0.9     # 引力尺度因子
        threshold_value = 3.0     # 阈值
        current_y, current_x = y, x   # 当前位置坐标,y是行，x是列
        distance = maze_distance[current_y, current_x]
        if distance >= threshold_value:
            pf_f = (pf_maze_size - pf_k * distance) * 0.9  # 引力
        else:
            pf_f = (pf_maze_size - pf_k * threshold_value) * 0.9

        # 根据目标的位置选择动作
        if current_x <= target_state_pos[1] and current_y > target_state_pos[0]:
            pf_action = 0   # 上
        elif current_x >= target_state_pos[1] and current_y < target_state_pos[0]:
            pf_action = 1   # 下
        elif current_x > target_state_pos[1] and current_y >= target_state_pos[0]:
            pf_action = 2   # 左
        elif current_x < target_state_pos[1] and current_y <= target_state_pos[0]:
            pf_action = 3   # 右
        return pf_f, pf_action  # 返回引力和动作

    # 存储记忆
    def remember(self, s, a, r, s_, done):
        # 存储正奖赏记忆
        if r > 0:
            self.memory1.append((s, a, r, s_, done))
            self.memory_count1 += 1
        else:
            self.memory2.append((s, a, r, s_, done))
            self.memory_count2 += 1

    # 记忆回放
    def repay(self):
        for i in range(batch_size):
            # rho=0.3的概论选择正奖赏记忆
            # 但rho不能太大，否则就不能探索出其他未知的路径
            if np.random.uniform() < rho and self.memory_count1 >= 1:
                mc1 = self.memory_count1
                # 防溢出
                if mc1 >= memory_size1:
                    mc1 = memory_size1
                num = np.random.choice(mc1, 1)  # 随机选择位置
                current_state, action, reward, next_state, done = self.memory1[num[0]]
            else:
                mc2 = self.memory_count2
                if mc2 >= memory_size2:
                    mc2 = memory_size2
                num = np.random.choice(mc2, 1)  # 随机选择位置
                current_state, action, reward, next_state, done = self.memory2[num[0]]

            # 类型转换
            current_state = torch.FloatTensor(current_state)
            action = torch.LongTensor([action])
            action = torch.unsqueeze(action, 0)
            reward = torch.FloatTensor([reward])
            next_state = torch.FloatTensor(next_state)
            # 到达终点
            if done:
                target = reward
            else:
                q_next = self.target_net(next_state).detach()   # 目标网络不进行反向传播
                target = reward + self.gamma * q_next.max(1)[0]     # 选出Qmax
            target = torch.unsqueeze(torch.FloatTensor(target), 0)  # 维数转换
            # 重复训练加快收敛
            for k in range(5):
                q_eval = self.eval_net(current_state)
                q_target = self.eval_net(current_state).detach()
                q_target[:, action] = target    # 把选择的动作对应的Q值换成target值
                loss = self.loss_func(q_eval, q_target)     # 计算损失
                self.optimizer.zero_grad()  # 梯度置零，不然会一直累加
                loss.backward()     # 反向传播
                self.optimizer.step()
        # “软”更新目标Q网络权重的参数  ω-←τω+（1-τ）ω-
        self.target_net.conv1.weight = nn.Parameter(self.eval_net.conv1.weight * tao + self.target_net.conv1.weight * (1 - tao))
        self.target_net.conv2.weight = nn.Parameter(self.eval_net.conv2.weight * tao + self.target_net.conv2.weight * (1 - tao))
        self.target_net.fc1.weight = nn.Parameter(self.eval_net.fc1.weight * tao + self.target_net.fc1.weight * (1 - tao))
        self.target_net.fc2.weight = nn.Parameter(self.eval_net.fc2.weight * tao + self.target_net.fc2.weight * (1 - tao))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


# 环境
class Environ:
    def __init__(self):
        pass

    # 根据当前状态current_state和动作action，返回next_state, reward, done
    def run(self, current_state, action):
        row, col = (current_state == POS_VALUE).nonzero()[0, 2:4]
        if action == 0:  # 0为'上'
            next_state_pos = (row - 1, col)
        elif action == 1:   # 1为‘下’
            next_state_pos = (row + 1, col)
        elif action == 2:   # 2为‘左’
            next_state_pos = (row, col - 1)
        else:   # 3为‘右’
            next_state_pos = (row, col + 1)

        if next_state_pos[0] < 0 or next_state_pos[0] >= maze.shape[0] \
                or next_state_pos[1] < 0 or next_state_pos[1] >= maze.shape[1]:
            # 如果出界，保持原地不动
            next_state = copy.deepcopy(current_state)
            reward = -1
            done = 0
            # print(next_state)
        elif maze[next_state_pos[0], next_state_pos[1]] == 1:
            # 如果遇到障碍物，保持不动
            next_state = copy.deepcopy(current_state)
            reward = -1     # 奖励为-1
            if test_flg == 0:   # 如果在训练阶段
                done = 0
            else:   # 如果在测试阶段，一遇到障碍物就结束
                done = 1
        elif next_state_pos == target_state_pos:
            # 如果到达目标
            next_state = matrix_to_img(target_state_pos[0], target_state_pos[1])
            row, col = target_state_pos[0], target_state_pos[1]
            reward = 1
            done = 1
        else:
            next_state = matrix_to_img(next_state_pos[0], next_state_pos[1])
            row, col = next_state_pos[0], next_state_pos[1]
            reward = 0
            done = 0
        return next_state, reward, done, row, col


def train_net():
    dqn.target_net.load_state_dict(dqn.eval_net.state_dict())
    episode = 1000  # 迭代次数
    step_count = 0
    environ = Environ()
    for ii in range(episode):
        current_state = matrix_to_img(start_state_pos[0], start_state_pos[1])
        current_row, current_col = start_state_pos[0], start_state_pos[1]
        j = 0
        step_count2 = 0
        while 1:
            j += 1
            action = dqn.choose_actions(current_state, current_row, current_col)  # 选择行为
            next_state, reward, done, current_row, current_col = environ.run(current_state, action)   # 在环境中施加行为推动游戏进行
            dqn.remember(current_state, action, reward, next_state, done)   # 记忆先前的状态，行为，奖励值与下一个状态
            step_count += 1
            step_count2 += 1
            # 游戏结束，跳出循环，进入下次迭代
            if done == 1:
                print("episode: {}, step used:{}".format(ii, j))
                episode_list.append(ii)
                step_list.append(j)
                break
            current_state = next_state
            # 积累10000步的经验和设置频率
            # 当网络的效果较好时就不去训练网络了
            # if step_count > 10000 and step_count2 % 100 == 0 and step_count2 > 50:
            if step_count > 10000 and step_count2 % 100 == 0:
                dqn.repay()
            if step_count % 300 == 0:
                # 更新目标网络
                dqn.target_net.load_state_dict(dqn.eval_net.state_dict())


def test_net():
    environ = Environ()
    global test_flg
    test_flg = 1
    current_state = matrix_to_img(start_state_pos[0], start_state_pos[1])
    for i in range(100):
        action_val = dqn.eval_net.forward(current_state)    #前向传播
        action = torch.max(action_val, 1)[1].data.numpy()[0]  # 选出最大值
        next_state, reward, done, _, _ = environ.run(current_state, action)
        print('current_state: {}, action: {}, next_state: {}'.format((current_state == POS_VALUE).nonzero()[0, 2:4],
                                                                     actions[action],
                                                                     (next_state == POS_VALUE).nonzero()[0, 2:4]))
        if done:
            break
        current_state = copy.deepcopy(next_state)


dqn = DQN()     # 定义 DQN 系统
start = time.clock()
train_net()     # 训练网络
test_net()      # 测试网络
elapsed = (time.clock() - start)
print("Time used:", elapsed)
fig = plt.figure(figsize = (10, 10))
plt.plot(episode_list, step_list)
plt.ylabel("step")
plt.xlabel("episode")
plt.title("DQN")
plt.ylim(0, 6000)
plt.xlim(0, 1000)
ax = plt.gca()
ax.xaxis.set_major_locator(tk.MultipleLocator(100))
ax.yaxis.set_major_locator(tk.MultipleLocator(500))
plt.show()



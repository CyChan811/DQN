clear all;
clc;
%% 地图map
global map;
    map = [
            0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
            0 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0
            0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0
            1 1 0 0 0 1 0 0 1 1 1 0 0 0 0 0
            0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0
            0 1 1 1 1 1 0 0 0 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0
            0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 1 1 1 0 0 1 1 1 0 0 0 0
            0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0                   
    ];

% map = [
%         0 0 0 0 0 0 0 0
%         1 1 1 1 1 1 1 0 
%         0 0 0 0 0 0 0 0 
%         0 1 1 1 1 1 1 1 
%         0 0 0 0 0 0 0 0 
%         1 1 1 1 1 1 1 0 
%         0 0 0 0 0 0 0 0
%         0 1 1 1 1 1 1 1
%         0 0 0 0 0 0 0 0 
%                         
% ];

% map = [
%        0 0 0 0 0 0 0 0 0 0 0 0 0
%        0 0 0 0 0 0 0 0 0 0 0 0 0
%        0 0 0 1 1 1 1 1 1 1 0 0 0
%        0 0 0 0 0 0 1 0 0 0 0 0 0
%        0 0 0 0 0 0 1 0 0 0 0 0 0
%        0 0 0 0 0 0 1 0 0 0 0 0 0
%        0 0 0 0 0 0 1 0 0 0 0 0 0
%        0 0 0 0 0 0 1 0 0 0 0 0 0
%        0 0 0 0 0 0 1 0 0 0 0 0 0
%        0 0 0 0 0 0 1 0 0 0 0 0 0
% ];

% map = [
%         0 0 0 0 0 0 0 0 1 0 0 0 0 0
%         0 1 1 1 1 1 0 0 1 0 0 0 0 0
%         0 0 0 0 0 1 0 0 1 0 0 0 0 0
%         1 1 0 0 0 1 0 0 1 1 1 0 0 0
%         0 0 0 0 0 1 0 0 1 0 1 0 0 0
%         0 1 1 1 1 1 0 0 0 0 1 0 0 0
%         0 0 0 0 0 0 0 1 0 0 0 0 0 0
%         0 0 0 0 0 1 1 1 0 0 0 0 0 0
%         0 0 0 0 0 0 0 1 1 1 1 1 1 1
%         0 0 0 0 0 1 0 0 0 0 0 1 0 0
%         0 0 0 1 1 1 1 0 0 1 1 1 0 0
%         0 0 0 0 0 1 0 0 0 0 0 0 0 0                
% ];

%  map = [
%          0 0 0 0 0 0 0 0
%          0 0 0 0 0 0 0 0 
%          0 0 1 1 1 0 0 0 
%          0 0 0 0 1 0 0 0 
%          0 0 0 0 1 0 0 0 
%          0 0 1 1 1 0 0 0 
%          0 0 0 0 0 0 0 0
%          0 0 0 0 0 0 0 0
%  
%  ];

% map = [
%        0 0 0 0 0
%        0 0 1 0 0
%        0 1 0 1 0
%        0 0 0 0 0
%        0 0 0 0 0
% ];
%% 参数初始化
action_num = 4;    %4个动作，上、下、左、右
state_num = numel(map); %获取地图状态数
start_state_pos = [1,1];   %起点
target_state_pos = [10,3]; %终点
actions = [1;2;3;4];    %上下左右
reward = [-1 0 1];
global POS_VALUE;  %当走到(row,col)时，令迷宫矩阵在(row,col)处的值为POS_VALUE
POS_VALUE = 2;
%定义全局变量
global dqn_net;
global target_net;
global batch_size; %数据集
global epsilon;   %动作探索值
global epsilon_min;   %epsilon最小
global epsilon_decay;   %探索衰减值
global gamma; %奖励递减参数
global tao;  %“软”更新目标Q网络权重的参数  ω-←τω+（1-τ）ω-
global memory_size1;  %正奖励序列记忆库大小  
global memory_count1;
global memory1;    %正奖励序列记忆库
global memory_size2;  %负奖励或零奖励序列记忆库大小
global memory_count2;
global memory2;    %负奖励或零奖励序列记忆库
global rho;   %以0.8概论从正奖励序列中抽取样本
global options;
global test_flg;
test_flg = 0;
%参数初始化
batch_size = 100; %数据集
epsilon = 1;   %动作探索值
epsilon_min = 0.1;   %epsilon最小
epsilon_decay = 0.99;   %探索衰减值
gamma = 0.9; %奖励递减参数
tao = 0.01;  %“软”更新目标Q网络权重的参数  ω-←τω+（1-τ）ω-
memory_size1 = 2000;  %正奖励序列记忆库大小  
memory_count1 = 0;
memory1 = {};    %正奖励序列记忆库
memory_size2 = 2000;  %负奖励或零奖励序列记忆库大小
memory_count2 = 0;
memory2 = {};    %负奖励或零奖励序列记忆库
rho = 0.3;   %以0.5概论从正奖励序列中抽取样本
tic;
dqn_net = dqn_model();  %创建神经网络  
options = trainingOptions('sgdm', ...
                'InitialLearnRate',0.1, ...%神经网络的学习速率设为0.01
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropFactor',0.1, ...  %更新学习速度
                'LearnRateDropPeriod',20, ...   %隔20次更新学习速度
                'MaxEpoch',30, ...  %重复训练同一批数据30次
                'MiniBatchSize',100);  
current_state = matrix_to_img(start_state_pos(1),start_state_pos(2));   %地图维数装换
dqn_net = trainNetwork(current_state, [0 0 0 0], dqn_net, options);
target_net = dqn_net;
train_net(actions,start_state_pos,target_state_pos);
toc;
state_mark = test_net(actions,start_state_pos,target_state_pos);
showmap(map,start_state_pos,target_state_pos,state_mark);  %显示地图
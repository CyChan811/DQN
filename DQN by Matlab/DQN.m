clear all;
clc;
%% ��ͼmap
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
%% ������ʼ��
action_num = 4;    %4���������ϡ��¡�����
state_num = numel(map); %��ȡ��ͼ״̬��
start_state_pos = [1,1];   %���
target_state_pos = [10,3]; %�յ�
actions = [1;2;3;4];    %��������
reward = [-1 0 1];
global POS_VALUE;  %���ߵ�(row,col)ʱ�����Թ�������(row,col)����ֵΪPOS_VALUE
POS_VALUE = 2;
%����ȫ�ֱ���
global dqn_net;
global target_net;
global batch_size; %���ݼ�
global epsilon;   %����̽��ֵ
global epsilon_min;   %epsilon��С
global epsilon_decay;   %̽��˥��ֵ
global gamma; %�����ݼ�����
global tao;  %��������Ŀ��Q����Ȩ�صĲ���  ��-���Ӧ�+��1-�ӣ���-
global memory_size1;  %���������м�����С  
global memory_count1;
global memory1;    %���������м����
global memory_size2;  %���������㽱�����м�����С
global memory_count2;
global memory2;    %���������㽱�����м����
global rho;   %��0.8���۴������������г�ȡ����
global options;
global test_flg;
test_flg = 0;
%������ʼ��
batch_size = 100; %���ݼ�
epsilon = 1;   %����̽��ֵ
epsilon_min = 0.1;   %epsilon��С
epsilon_decay = 0.99;   %̽��˥��ֵ
gamma = 0.9; %�����ݼ�����
tao = 0.01;  %��������Ŀ��Q����Ȩ�صĲ���  ��-���Ӧ�+��1-�ӣ���-
memory_size1 = 2000;  %���������м�����С  
memory_count1 = 0;
memory1 = {};    %���������м����
memory_size2 = 2000;  %���������㽱�����м�����С
memory_count2 = 0;
memory2 = {};    %���������㽱�����м����
rho = 0.3;   %��0.5���۴������������г�ȡ����
tic;
dqn_net = dqn_model();  %����������  
options = trainingOptions('sgdm', ...
                'InitialLearnRate',0.1, ...%�������ѧϰ������Ϊ0.01
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropFactor',0.1, ...  %����ѧϰ�ٶ�
                'LearnRateDropPeriod',20, ...   %��20�θ���ѧϰ�ٶ�
                'MaxEpoch',30, ...  %�ظ�ѵ��ͬһ������30��
                'MiniBatchSize',100);  
current_state = matrix_to_img(start_state_pos(1),start_state_pos(2));   %��ͼά��װ��
dqn_net = trainNetwork(current_state, [0 0 0 0], dqn_net, options);
target_net = dqn_net;
train_net(actions,start_state_pos,target_state_pos);
toc;
state_mark = test_net(actions,start_state_pos,target_state_pos);
showmap(map,start_state_pos,target_state_pos,state_mark);  %��ʾ��ͼ
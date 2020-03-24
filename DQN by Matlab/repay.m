%从记忆容器memory中随机选择(current_state, action, reward, next_state, done),然后送入模型进行训练
function repay(batch_size)
    global map;
    global dqn_net;
    global target_net;
    global rho;
    global memory_count1;
    global memory_count2;
    global memory1;
    global memory2;
    global tao;
    global gamma;
    global epsilon;
    global epsilon_min;
    global epsilon_decay;
    global options;

    batch_current_state = zeros(size(map,1),size(map,2),1,batch_size);
    batch_target = zeros(batch_size,4);
    random_chose_memory = cell(batch_size, 5);
    for i = 1:batch_size
        if rand() < rho && memory_count1 > 0    %正奖赏
            mc1 = memory_count1;
            if mc1 > 2000
               mc1 = 2000; 
            end
            num = randperm(mc1, 1);
            random_chose_memory{i,1} = memory1{num(1),1}; %随机选取current_state样本数据
            random_chose_memory{i,2} = memory1{num(1),2}; %随机选取action样本数据
            random_chose_memory{i,3} = memory1{num(1),3}; %随机选取reward样本数据
            random_chose_memory{i,4} = memory1{num(1),4}; %随机选取next_state样本数据
            random_chose_memory{i,5} = memory1{num(1),5}; %随机选取done样本数据
       % random_chose_memory = this.memory1{randperm(this.memory_count1, batch_size),:}; %随机选取样本数据
        else    %零或负奖赏
            mc2 = memory_count2;
            if mc2 > 2000
               mc2 = 2000; 
            end
            num = randperm(mc2, 1);
            random_chose_memory{i,1} = memory2{num(1),1}; %随机选取样本数据
            random_chose_memory{i,2} = memory2{num(1),2}; %随机选取样本数据
            random_chose_memory{i,3} = memory2{num(1),3}; %随机选取样本数据
            random_chose_memory{i,4} = memory2{num(1),4}; %随机选取样本数据
            random_chose_memory{i,5} = memory2{num(1),5}; %随机选取样本数据
            %random_chose_memory = this.memory2{randperm(this.memory_count2, batch_size),:};
        end
        current_state = random_chose_memory{i,1};
        batch_current_state(:,:,:,i) = current_state;
        action = random_chose_memory{i,2};
        reward = random_chose_memory{i,3};
        next_state = random_chose_memory{i,4};
        done = random_chose_memory{i,5};
        target_f = predict(dqn_net, current_state);

        if done
           target = reward;
        else
            target = predict(target_net, next_state);
            target = reward + gamma .* max(target); %更新公式
        end
         target_f(1,action) = target;
         batch_target(i,:) = target_f;
    end
    dqn_net = trainNetwork(batch_current_state, batch_target, dqn_net.Layers, options);
    %更新探索值
    if epsilon > epsilon_min
       epsilon = epsilon * epsilon_decay;
    else
        epsilon = epsilon_min;
    end
end







%�Ӽ�������memory�����ѡ��(current_state, action, reward, next_state, done),Ȼ������ģ�ͽ���ѵ��
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
        if rand() < rho && memory_count1 > 0    %������
            mc1 = memory_count1;
            if mc1 > 2000
               mc1 = 2000; 
            end
            num = randperm(mc1, 1);
            random_chose_memory{i,1} = memory1{num(1),1}; %���ѡȡcurrent_state��������
            random_chose_memory{i,2} = memory1{num(1),2}; %���ѡȡaction��������
            random_chose_memory{i,3} = memory1{num(1),3}; %���ѡȡreward��������
            random_chose_memory{i,4} = memory1{num(1),4}; %���ѡȡnext_state��������
            random_chose_memory{i,5} = memory1{num(1),5}; %���ѡȡdone��������
       % random_chose_memory = this.memory1{randperm(this.memory_count1, batch_size),:}; %���ѡȡ��������
        else    %��򸺽���
            mc2 = memory_count2;
            if mc2 > 2000
               mc2 = 2000; 
            end
            num = randperm(mc2, 1);
            random_chose_memory{i,1} = memory2{num(1),1}; %���ѡȡ��������
            random_chose_memory{i,2} = memory2{num(1),2}; %���ѡȡ��������
            random_chose_memory{i,3} = memory2{num(1),3}; %���ѡȡ��������
            random_chose_memory{i,4} = memory2{num(1),4}; %���ѡȡ��������
            random_chose_memory{i,5} = memory2{num(1),5}; %���ѡȡ��������
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
            target = reward + gamma .* max(target); %���¹�ʽ
        end
         target_f(1,action) = target;
         batch_target(i,:) = target_f;
    end
    dqn_net = trainNetwork(batch_current_state, batch_target, dqn_net.Layers, options);
    %����̽��ֵ
    if epsilon > epsilon_min
       epsilon = epsilon * epsilon_decay;
    else
        epsilon = epsilon_min;
    end
end







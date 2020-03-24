function state_mark = test_net(actions, start_state_pos, target_state_pos)
    global dqn_net;
    global POS_VALUE;
    global test_flg;
    test_flg = 1;
    current_state = matrix_to_img(start_state_pos(1), start_state_pos(2));
    state_mark = zeros(100,2);
    %最多走500步，超过500游戏结束
    for i = 1:100
       %kk = randperm(1);
       %current_state = current_state(:, :, kk((1 - 1) * 1 + 1 : 1 * 1));
       action = predict(dqn_net, current_state);
       action_ = find(action == max(action)); %action最大值的索引 即为要执行的下一个动作
       [next_state,reward,done] = step(current_state, action_, target_state_pos);    %在环境中施加行为推动游戏进行
       [current_state_row,current_state_column] = find(current_state == POS_VALUE);
       state_mark(i,:) = [current_state_row,current_state_column];
       %fprintf('current_state:%d %d,action: %d,next_state:%d %d',current_state_row,current_state_column, action_, next_state_row,next_state_column);
       %如果游戏结束，跳出循环
        if (done == 1)
            if i < 99
                [current_state_row,current_state_column] = find(next_state == POS_VALUE);
                state_mark(i+1,:) = [current_state_row,current_state_column];
                state_mark(i+2:100,:) = [];
            elseif i == 99
                [current_state_row,current_state_column] = find(next_state == POS_VALUE);
                state_mark(i+1,:) = [current_state_row,current_state_column];
            end
            break;
        end
        %使下一个状态成为下一帧的新状态
        current_state = next_state;
    end
end
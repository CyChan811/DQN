function state_mark = test_net(actions, start_state_pos, target_state_pos)
    global dqn_net;
    global POS_VALUE;
    global test_flg;
    test_flg = 1;
    current_state = matrix_to_img(start_state_pos(1), start_state_pos(2));
    state_mark = zeros(100,2);
    %�����500��������500��Ϸ����
    for i = 1:100
       %kk = randperm(1);
       %current_state = current_state(:, :, kk((1 - 1) * 1 + 1 : 1 * 1));
       action = predict(dqn_net, current_state);
       action_ = find(action == max(action)); %action���ֵ������ ��ΪҪִ�е���һ������
       [next_state,reward,done] = step(current_state, action_, target_state_pos);    %�ڻ�����ʩ����Ϊ�ƶ���Ϸ����
       [current_state_row,current_state_column] = find(current_state == POS_VALUE);
       state_mark(i,:) = [current_state_row,current_state_column];
       %fprintf('current_state:%d %d,action: %d,next_state:%d %d',current_state_row,current_state_column, action_, next_state_row,next_state_column);
       %�����Ϸ����������ѭ��
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
        %ʹ��һ��״̬��Ϊ��һ֡����״̬
        current_state = next_state;
    end
end
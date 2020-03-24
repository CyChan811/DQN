% ���ݵ�ǰ״̬current_state�Ͷ���action������next_state, reward, done
function [next_state, reward, done] = step(current_state, action, target_state_pos)
    %% ����
    global POS_VALUE;
    global map;
    global test_flg;
    [row,col] = find(current_state == POS_VALUE);
    done = 0;
    if action == 1  % 1 = ��up��
        next_state_pos = [row-1,col];
    elseif action == 2 % 2 = ��down��
        next_state_pos = [row+1,col];
    elseif action == 3 % 3 = 'left'
        next_state_pos = [row,col-1];
    else % 'right'
        next_state_pos = [row,col+1];
    end

    if next_state_pos(1) < 1 || next_state_pos(1) > size(map,1) ...
            || next_state_pos(2) < 1 || next_state_pos(2) > size(map,2)
        % ������磬����ԭ�ز���
        next_state = current_state
        reward = -1;
        done = 0;
    elseif map(next_state_pos(1),next_state_pos(2)) == 1    %�����ϰ���
        next_state = current_state
        reward = -1;
        if test_flg == 0    %ѵ���׶�
            done = 0;   
        elseif test_flg == 1    %����ڲ��Խ׶�
            done = 1;  %�����ϰ������
        end
    elseif (next_state_pos(1)==target_state_pos(1) && next_state_pos(2)==target_state_pos(2))    % ����Ŀ��
        next_state = matrix_to_img(target_state_pos(1), target_state_pos(2))
        reward = 1;
        done = 1;   %����Ŀ�����¿�ʼ
    else 
        next_state = matrix_to_img(next_state_pos(1), next_state_pos(2))
        reward = 0;
        done = 0;
    end
end
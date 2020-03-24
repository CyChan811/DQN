% �˹��Ƴ�,���������Ͷ���
function [pf_f, pf_action] = potential_field(y, x, target_state_pos)
    global map;
    pf_maze_size = sqrt(power(size(map, 1), 2) + power(size(map, 2), 2));   %��ͼб����
    pf_k = 0.9; %�����߶�����
    threshold_value = 3.0;     % ��ֵ
    current_y = y;   % ��ǰλ������,y���У�x����
    current_x = x;
    distance = sqrt(power((target_state_pos(1) - current_y), 2) + power((target_state_pos(2) - current_x), 2));
    if distance >= threshold_value
        pf_f = (pf_maze_size - pf_k * distance) * 0.9;  % ����
    else
        pf_f = (pf_maze_size - pf_k * threshold_value) * 0.9;
    end
    % ����Ŀ���λ��ѡ����
    if current_x <= target_state_pos(2) && current_y > target_state_pos(1)
        pf_action = 1;   % ��
    elseif current_x >= target_state_pos(2) && current_y < target_state_pos(1)
        pf_action = 2;   % ��
    elseif current_x > target_state_pos(2) && current_y >= target_state_pos(1)
        pf_action = 3;   % ��
    elseif current_x < target_state_pos(2) && current_y <= target_state_pos(1)
        pf_action = 4;   % ��
    end
end
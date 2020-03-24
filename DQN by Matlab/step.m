% 根据当前状态current_state和动作action，返回next_state, reward, done
function [next_state, reward, done] = step(current_state, action, target_state_pos)
    %% 环境
    global POS_VALUE;
    global map;
    global test_flg;
    [row,col] = find(current_state == POS_VALUE);
    done = 0;
    if action == 1  % 1 = ‘up’
        next_state_pos = [row-1,col];
    elseif action == 2 % 2 = ‘down’
        next_state_pos = [row+1,col];
    elseif action == 3 % 3 = 'left'
        next_state_pos = [row,col-1];
    else % 'right'
        next_state_pos = [row,col+1];
    end

    if next_state_pos(1) < 1 || next_state_pos(1) > size(map,1) ...
            || next_state_pos(2) < 1 || next_state_pos(2) > size(map,2)
        % 如果出界，保持原地不动
        next_state = current_state
        reward = -1;
        done = 0;
    elseif map(next_state_pos(1),next_state_pos(2)) == 1    %遇到障碍物
        next_state = current_state
        reward = -1;
        if test_flg == 0    %训练阶段
            done = 0;   
        elseif test_flg == 1    %如果在测试阶段
            done = 1;  %遇到障碍物结束
        end
    elseif (next_state_pos(1)==target_state_pos(1) && next_state_pos(2)==target_state_pos(2))    % 到达目标
        next_state = matrix_to_img(target_state_pos(1), target_state_pos(2))
        reward = 1;
        done = 1;   %到达目标重新开始
    else 
        next_state = matrix_to_img(next_state_pos(1), next_state_pos(2))
        reward = 0;
        done = 0;
    end
end
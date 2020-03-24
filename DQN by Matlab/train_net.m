function train_net(actions, start_state_pos, target_state_pos)
    global target_net;
    global dqn_net;
    global tao;
    episode = 1000;   %��������
    fid = fopen('test.txt','w');
    h=waitbar(0,'the net is training...');  %������
    step_count = 0;
    for ii = 1:episode
        current_state = matrix_to_img(start_state_pos(1),start_state_pos(2));
        j = 1;
        waitbar(ii/episode);
        while (1)
           j = j + 1;
           action = choose_action(actions, current_state,target_state_pos); %ѡ����Ϊ
           [next_state,reward,done] = step(current_state, action, target_state_pos); %�ڻ�����ʩ����Ϊ�ƶ���Ϸ����
           remember(current_state, action, reward, next_state, done); %������ǰ��״̬����Ϊ������ֵ����һ��״̬
           step_count = step_count + 1;
           if (done == 1)
                fprintf(fid,'episode: %d, step used: %d\n',ii, j);
                break;
           end
           %��Ϸ����������ѭ���������´ε���
           if step_count > 10000 && rem(step_count,100) == 0  %���۾��������Ƶ��
                repay(100);
           end
           current_state = next_state;
           if rem(step_count, 400) == 0     %����Ŀ������
               target_net = dqn_net;
           end
        end
    end
    close(h);
    fclose(fid);
end
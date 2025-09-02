clear;
sum_loss = double(0);
for i = 1:54
    if i < 10
        idx = strcat('0', string(i));
    else
        idx = string(i);
    end
    % ��ȡ���ݼ�
    str = sprintf('D:/MI-BCI/openBMI/Dataset/session_1/s%d/sess01_subj%s_EEG_MI.mat', i, idx);
    [CNT_tr,CNT_te] = Load_MAT(str);
    % ѵ����Ԥ����
    CNT_tr = prep_selectChannels(CNT_tr,{'Name', {'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'}});
    CNT_tr = prep_filter(CNT_tr,{'frequency',[8,30]});%����Ƶ��8��30����
    original_signal = CNT_tr.smt; % ԭʼ�źţ���СΪ 4000 �� 100 �� 62
    original_fs = 1000; % ԭʼ�����ʣ���λΪHz
    target_fs = 250; % Ŀ������ʣ���λΪHz
    factor = original_fs / target_fs;
    [time_points, trials, channels] = size(original_signal);
    downsampled_signal = zeros(floor(time_points / factor), trials, channels);
    for ch = 1:channels
        for tr = 1:trials
            downsampled_signal(:, tr, ch) = decimate(original_signal(:, tr, ch), factor);
        end
    end
    downsampled_signal_1 = downsampled_signal(:, :, 8:11);
    downsampled_signal_2 = downsampled_signal(:, :, 13:15);
    downsampled_signal_3 = downsampled_signal(:, :, 18:21);
    downsampled_signal_4 = downsampled_signal(:, :, 33:41);
    downsampled_signal_tr = cat(3,downsampled_signal_1,downsampled_signal_2,downsampled_signal_3,downsampled_signal_4);
    label_tr = CNT_tr.y_dec;
    label_tr = label_tr';
%�õ�SMT��һ������
%���Լ�Ԥ����
    CNT_te = prep_selectChannels(CNT_te,{'Index',1:20});
    CNT_te = prep_filter(CNT_te,{'frequency',[8,30]});%����Ƶ��8��30����
    original_signal = CNT_te.smt; % ԭʼ�źţ���СΪ 4000 �� 100 �� 62
    original_fs = 1000; % ԭʼ�����ʣ���λΪHz
    target_fs = 250; % Ŀ������ʣ���λΪHz
    factor = original_fs / target_fs;
    [time_points, trials, channels] = size(original_signal);
    downsampled_signal = zeros(floor(time_points / factor), trials, channels);
    for ch = 1:channels
        for tr = 1:trials
            downsampled_signal(:, tr, ch) = decimate(original_signal(:, tr, ch), factor);
        end
    end
    label_te = CNT_te.y_dec;
    label_te = label_te';
    label=cat(1,label_tr,label_te); %���ձ�ǩ
    
    downsampled_signal_te = downsampled_signal(:,:,1:20);
    data = cat(2,downsampled_signal_tr,downsampled_signal_te); %��������
    
    
    saveDir = ['D:/MI-BCI/openBMI/data_train/s',num2str(i),'T.mat'];
    save(saveDir,'data','label');
end








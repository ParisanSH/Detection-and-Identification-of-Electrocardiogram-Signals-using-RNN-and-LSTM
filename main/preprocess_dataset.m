clc;
clear;
%%
S = cell(73,1);ant = cell(73,1);
for i= 1:73
   f = strcat('D:\NN HW 2021\HW3\dataset\samples_',num2str(i),'.csv');
   S{i,1} = readtable(f);
end
for i= 1:73
   f = strcat('D:\NN HW 2021\HW3\dataset\annotation_',num2str(i),'.csv');
   ant{i,1} = readtable(f);
end
samp= zeros(15000,45);
samp2 = zeros(2500,28);
for i=1:45
    t = S{i}{2:end,2};
    samp(:,i) = str2double(t);
end
for i=46:73
    t = S{i}{2:end,2};
    samp2(:,i-45) = str2double(t);
end
samp(:,15)=[];
ant1 = zeros(15000,45);
ant2 = zeros(2500,28);
for i=1:45
    if i~=15
        t = table2array(ant{i});
        ant1(:,i)= t;  
    end
end
ant1(:,15)=[];
for i=46:73
    t = table2array(ant{i});
    ant2(:,i-45)= t;  
end
%%
train = [];test = [];
window = 11; % 11, 21, 5
split_1 = round((15000-window)* 0.8);
split_2 = round((2500-window)* 0.8);
for i=1:44
    smp = samp(:,i);
    ant = ant1(:,i);
    for j=1:15000-window
        t = zeros(window*2,1);
        for k=1:window
           t(2*k-1,1)=smp(j+k-1);
           t(2*k,1)=ant(j+k-1);
        end
        if j<=split_1
            train=[train,t];
        else
            test =[test,t];
        end
    end
end
for i=1:28
    smp = samp2(:,i);
    ant = ant2(:,i);
    for j=1:2500-window
        t = zeros(window*2,1); % 22 ,42 , 10
        for k=1:window %11 , 21 , 5
           t(2*k-1,1)=smp(j+k-1);
           t(2*k,1)=ant(j+k-1);
        end
        if j<=split_2
            train=[train,t];
        else
            test =[test,t];
        end
    end
end
target = train(window*2,:);
train(window*2,:)=[];
save('train21.mat','train')
save('target21.mat','target')
save('test21.mat','test')
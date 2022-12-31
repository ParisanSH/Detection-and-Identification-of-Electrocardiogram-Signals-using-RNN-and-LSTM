clc;clear;close;
workspace;format longg;format compact;
start_path = fullfile(matlabroot,'\toolbox\images\imdemos');
p = uigetdir(start_path);
if p == 0
	return;
end
slidong_win = 11; % 11 , 5 , 21
samples1_45= zeros(15000,45);
samples46_73 = zeros(2500,28);
annotations1_45 = zeros(15000,45);
annotations46_73 = zeros(2500,28);
for i= 1:45
   filetable = strcat(p,'\samples_',num2str(i),'.csv');
   temp = readtable(filetable);
   temp = str2double(temp{2:end,2});
   samples1_45(:,i) = temp;
   filetable = strcat(p,'\annotation_',num2str(i),'.csv');
   if i~=15
       temp = table2array(readtable(filetable));
       annotations1_45(:,i)= temp;  
   end
end
for i= 46:73
   filetable = strcat(p,'\samples_',num2str(i),'.csv');
   temp = readtable(filetable);
   temp = str2double(temp{2:end,2});
   samples46_73(:,i-45) = temp;
   filetable = strcat(p,'\annotation_',num2str(i),'.csv');
   temp = table2array(readtable(filetable));
   annotations46_73(:,i-45)= temp;
end
samples1_45(:,15)=[];
annotations1_45(:,15)=[];
TrainDs = [];
TestDs = [];
t1 = round((15000-slidong_win)*0.8); 
t2 = round((2500-slidong_win)*0.8);

for i=1:44
smp = samples1_45(:,i);
ant = annotations1_45(:,i);
for j=1:15000-slidong_win %11 , 21 , 5
   temp = zeros(slidong_win*2,1); % 22 ,42 , 10
   for k=1:slidong_win %11 , 21 , 5
      temp(2*k-1,1)=smp(j+k-1);
      temp(2*k,1)=ant(j+k-1);
   end
   if j<=t1
      TrainDs=[TrainDs,temp];
   else
      TestDs =[TestDs,temp];
   end
end
end
for i=1:28
smp = samples46_73(:,i);
ant = annotations46_73(:,i);
for j=1:2500-slidong_win %11 , 21 , 5
   temp = zeros(slidong_win*2,1); % 22 ,42 , 10
   for k=1:slidong_win %11 , 21 , 5
      temp(2*k-1,1)=smp(j+k-1);
      temp(2*k,1)=ant(j+k-1);
   end
   if j<=t2
      TrainDs=[TrainDs,temp];
   else
      TestDs =[TestDs,temp];
   end
end
end
save('TrainDs_11.mat','TrainDs')
save('TestDs_11.mat','TestDs')
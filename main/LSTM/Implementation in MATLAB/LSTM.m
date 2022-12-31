%Farshid Pirbonyeh_______40033608
%NNDL_Prj_#4
clc
clear 
close all
%EXP
load HumanActivityTrain
%% Importing Stage ?????
%
%Selecting all csv files
[filename,filedir] = uigetfile('*.csv','Multiselect','on');
path = fullfile(filedir,filename);
path = path';
file = struct('name',path);
%
% Train Annotation 
for i=1:1:60
nam=filename(1,i);
addr=strcat('G:\Arshad\TERM2\Deep\Prj#4\dataset\',nam);
anno_in=readmatrix(char(addr));
anno=((anno_in(:,1))+1)';

my_field = strcat('annotation',num2str(i));
Train_Annotations.(my_field)=anno;
end


% Train Annotation 
for i=61:1:72
    j=i-60;
nam=filename(1,i);
addr=strcat('G:\Arshad\TERM2\Deep\Prj#4\dataset\',nam);
anno_in=readmatrix(char(addr));
anno=((anno_in(:,1))+1)';

my_field = strcat('annotation',num2str(j));
Test_Annotations.(my_field)=anno;
end


%Train Samples
for i=73:1:132
    j=i-72;
nam=filename(1,i);
addr=strcat('G:\Arshad\TERM2\Deep\Prj#4\dataset\',nam);
samp_in=readmatrix(char(addr));
samp=(samp_in(:,2))';

my_field = strcat('samples',num2str(j));
Train_Samples.(my_field)=samp;
end

%Test Samples
for i=133:1:144
    j=i-132;
nam=filename(1,i);
addr=strcat('G:\Arshad\TERM2\Deep\Prj#4\dataset\',nam);
samp_in=readmatrix(char(addr));
samp=(samp_in(:,2))';

my_field = strcat('samples',num2str(j));
Test_Samples.(my_field)=samp;
end
%% Spiliting to 500 Datas ????
% All Trains Annotations
i=1;
for j=1:1:length(fieldnames(Train_Annotations))
    my_field = strcat('annotation',num2str(j));
    A=Train_Annotations.(my_field);
    start=1;
    ending=500;
    o=(length(A)/500);
    for k=1:1:o
    B=A(1,start:ending);
    All_Train_Annotation{i,:}=B;
    start=start+500;
    ending=ending+500;
    i=i+1;
    end
end
% All Test Annotations
i=1;
for j=1:1:length(fieldnames(Test_Annotations))
    my_field = strcat('annotation',num2str(j));
    A=Test_Annotations.(my_field);
    start=1;
    ending=500;
    o=(length(A)/500);
    for k=1:1:o
    B=A(1,start:ending);
    All_Test_Annotation{i,:}=B;
    start=start+500;
    ending=ending+500;
    i=i+1;
    end
end
% All Trains Samples
i=1;
for j=1:1:length(fieldnames(Train_Samples))
    my_field = strcat('samples',num2str(j));
    A=Train_Samples.(my_field);
    start=1;
    ending=500;
    o=(length(A)/500);
    for k=1:1:o
    B=A(1,start:ending);
    All_Train_Samples{i,:}=B;
    start=start+500;
    ending=ending+500;
    i=i+1;
    end
end
% All Test Samples
i=1;
for j=1:1:length(fieldnames(Test_Samples))
    my_field = strcat('samples',num2str(j));
    A=Test_Samples.(my_field);
    start=1;
    ending=500;
    o=(length(A)/500);
    for k=1:1:o
    B=A(1,start:ending);
    All_Test_Samples{i,:}=B;
    start=start+500;
    ending=ending+500;
    i=i+1;
    end
end
    
 %% Windowing the Datas by 5 Data
 %Train
for l=1:1:length((All_Train_Samples)) 
        A=All_Train_Samples{l,1};
        B=All_Train_Annotation{l,1};
%A=1:20;
%B=31:51;
NUM_window=10;
C=zeros(1, NUM_window);
I=1;
SIZE=5;
for k=1:(numel(A)-4)
    IN=A(1,k:SIZE);
    OOO=B(1,k:SIZE);
    SIZE=SIZE+1;
j=1;
TWOxSIZE=10;
for i=2:2:TWOxSIZE
    a=IN(1,j);
    b=OOO(1,j);
    C(1,i-1)=a;
    C(1,i)=b;
    j=j+1;
end
   Input{:,I}=C(1,1:9)';
   Output(1,I)=C(1,10);
   I=I+1;
end
Windowed_5_TRXs{l,1}=cell2mat(Input);
Windowed_5_TRYs{l,1}=categorical(Output);
end
%Test

 R=1;
for l=1:1:length((All_Test_Samples)) 
        A=All_Test_Samples{l,1};
        B=All_Test_Annotation{l,1};
%A=1:20;
%B=31:51;
NUM_window=10;
C=zeros(1, NUM_window);
I=1;
SIZE=5;
for k=1:(numel(A)-4)
    IN=A(1,k:SIZE);
    OOO=B(1,k:SIZE);
    SIZE=SIZE+1;
j=1;
TWOxSIZE=10;
for i=2:2:TWOxSIZE
    a=IN(1,j);
    b=OOO(1,j);
    C(1,i-1)=a;
    C(1,i)=b;
    j=j+1;
end
   Input{:,I}=C(1,1:9)';
   Output(1,I)=C(1,10);
   I=I+1;
end
Windowed_5_TEXs{l,1}=cell2mat(Input);
Windowed_5_TEYs{l,1}=categorical(Output);
end


%%
%The
numFeatures = TWOxSIZE-1;
numHiddenUnits = 16;
numClasses = 4;

%layer = bilstmLayer(10,'OutputMode','sequence');
layers = [
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence');
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'Minibatchsize',4 ,...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',5, ...
    'shuffle','never',...
    'VerboseFrequency',2,...
    'Plots','training-progress');
net_5 = trainNetwork(Windowed_5_TRXs,Windowed_5_TRYs,layers,options);

%% Other Windows 
clear IN OOO  a b A B C Input Output
% Windowing the Datas by 10 Data
SIZE=10;
Datas_in_window=2*SIZE;% size of window x 2
unuse=SIZE-1;
Datas_of_in=Datas_in_window-1;
 
 %Train
 
 
for l=1:1:length((All_Train_Samples)) 
        A=All_Train_Samples{l,1};
        B=All_Train_Annotation{l,1};

C=zeros(1, Datas_in_window);
I=1;
sizez=10;
for k=1:(numel(A)-unuse)
    IN=A(1,k:sizez);
    OOO=B(1,k:sizez);
    sizez=sizez+1;
j=1;

for i=2:2:Datas_in_window
    a=IN(1,j);
    b=OOO(1,j);
    C(1,i-1)=a;
    C(1,i)=b;
    j=j+1;
end


   Input{:,I}=C(1,1:Datas_of_in)';
   Output(1,I)=C(1,Datas_in_window);
   I=I+1;
end
Windowed_10_TRXs{l,1}=cell2mat(Input);
Windowed_10_TRYs{l,1}=categorical(Output);
end
%Test

for l=1:1:length((All_Test_Samples)) 
        A=All_Test_Samples{l,1};
        B=All_Test_Annotation{l,1};
C=zeros(1, Datas_in_window);
I=1;
sizez=10;
for k=1:(numel(A)-unuse)
    IN=A(1,k:sizez);
    OOO=B(1,k:sizez);
    sizez=sizez+1;
j=1;

for i=2:2:Datas_in_window
    a=IN(1,j);
    b=OOO(1,j);
    C(1,i-1)=a;
    C(1,i)=b;
    j=j+1;
end
  
   Input{:,I}=C(1,1:Datas_of_in)';
   Output(1,I)=C(1,Datas_in_window);
   I=I+1;
end
Windowed_10_TEXs{l,1}=cell2mat(Input);
Windowed_10_TEYs{l,1}=categorical(Output);
end

numFeatures = Datas_of_in;
numHiddenUnits = 16;
numClasses = 4;

%layer = bilstmLayer(10,'OutputMode','sequence');
layers = [
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence');
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'Minibatchsize',4 ,...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',5, ...
    'shuffle','never',...
    'VerboseFrequency',2,...
    'Plots','training-progress');
net_10 = trainNetwork(Windowed_10_TRXs,Windowed_10_TRYs,layers,options);






clear IN OOO  a b A B C Input Output
% Windowing the Datas by 15 Data
SIZE=15;
Datas_in_window=2*SIZE;% size of window x 2
unuse=SIZE-1;
Datas_of_in=Datas_in_window-1;
 
 %Train
 
for l=1:1:length((All_Train_Samples)) 
        A=All_Train_Samples{l,1};
        B=All_Train_Annotation{l,1};

C=zeros(1, Datas_in_window);
I=1;
sizez=15;
for k=1:(numel(A)-unuse)
    IN=A(1,k:sizez);
    OOO=B(1,k:sizez);
    sizez=sizez+1;
j=1;

for i=2:2:Datas_in_window
    a=IN(1,j);
    b=OOO(1,j);
    C(1,i-1)=a;
    C(1,i)=b;
    j=j+1;
end


   Input{:,I}=C(1,1:Datas_of_in)';
   Output(1,I)=C(1,Datas_in_window);
   I=I+1;
end
Windowed_15_TRXs{l,1}=cell2mat(Input);
Windowed_15_TRYs{l,1}=categorical(Output);
end
%Test


for l=1:1:length((All_Test_Samples)) 
        A=All_Test_Samples{l,1};
        B=All_Test_Annotation{l,1};
C=zeros(1, Datas_in_window);
I=1;
sizez=15;
for k=1:(numel(A)-unuse)
    IN=A(1,k:sizez);
    OOO=B(1,k:sizez);
    sizez=sizez+1;
j=1;

for i=2:2:Datas_in_window
    a=IN(1,j);
    b=OOO(1,j);
    C(1,i-1)=a;
    C(1,i)=b;
    j=j+1;
end
  
   Input{:,I}=C(1,1:Datas_of_in)';
   Output(1,I)=C(1,Datas_in_window);
   I=I+1;
end
Windowed_15_TEXs{l,1}=cell2mat(Input);
Windowed_15_TEYs{l,1}=categorical(Output);
end

numFeatures = Datas_of_in;
numHiddenUnits = 16;
numClasses = 4;

%layer = bilstmLayer(10,'OutputMode','sequence');
layers = [
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence');
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'Minibatchsize',4 ,...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',5, ...
    'shuffle','never',...
    'VerboseFrequency',2,...
    'Plots','training-progress');
net_15 = trainNetwork(Windowed_15_TRXs,Windowed_15_TRYs,layers,options);

%% 
% Testing Phase

miniBatchSize=4;
Yout_from_5=classify(net_5,Windowed_5_TEXs,...
    'MiniBatchSize',miniBatchSize);
Yout_from_10=classify(net_10,Windowed_10_TEXs,...
    'MiniBatchSize',miniBatchSize);
Yout_from_15=classify(net_15,Windowed_15_TEXs,...
    'MiniBatchSize',miniBatchSize);

for i=1:60
    %Ins
    Y5_I(i,:)=double(Windowed_5_TEYs{i,1});
    Y10_I(i,:)=double(Windowed_10_TEYs{i,1});
    Y15_I(i,:)=double(Windowed_15_TEYs{i,1});
   
    %Outs
    Y5_O(i,:)=(double(Yout_from_5{i,1}));
    Y10_O(i,:)=(double(Yout_from_10{i,1}));
    Y15_O(i,:)=(double(Yout_from_15{i,1}));
end




Acc=100.*(sum(sum(Y5_O==Y5_I))./numel(Y5_O));
fprintf('\n\nOur Accurcy with Window Size 5 Will be: %2.2f %% \n\n', Acc)


Acc=100.*(sum(sum(Y10_O==Y10_I))./numel(Y10_O));
fprintf('\n\nOur Accurcy with Window Size 10  Will be: %2.2f %% \n\n', Acc)


Acc=100.*(sum(sum(Y15_O==Y15_I))./numel(Y15_O));
fprintf('\n\nOur Accurcy with Window Size 15 Will be: %2.2f %% \n\n', Acc)

  
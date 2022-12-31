clc;
clear;
close;
% load train and test datasets mat file
example = matfile('TrainDs_5.mat');
train_ = example.trainx;
example = matfile('TestDs_5.mat');
test = example.data_test;
% split samples and targets from train and test matrix
ytr = train_(42,:);
train_(42,:)= [];
yts = test(42,:);
test(42,:) = [];
%create and simulate elman network 
E_net=elmannet(1:2,15,'traingdx');
E_net.trainParam.epochs=300;
E_net=init(E_net);
[E_net,per] = train(E_net,train_,ytr); 
train_ty = sim(E_net, train_);
test_ty = sim(E_net, test);
ytr_pred = round(train_ty);
yts_pred = round(test_ty);
%compute elman accuracy 
trainAcc =0;testAcc =0;
for i=1:length(ytr)
    if ytr_pred(1,i) == ytr(1,i)
        trainAcc = trainAcc +1;
    end
    if i <= length(yts)
        if yts_pred(1,i) == yts(1,i)
        testAcc = testAcc +1;end
    end
end
%display elman results
fprintf('\t Elman network accuracy for train and test:\n')
fprintf('train acc =\t%f\n', trainAcc/length(ytr))
fprintf('test acc = \t%f\n', testAcc/length(yts))
figure(1)
x=1:length(train_ty);
plot(x(1:1000),ytr(1:1000),'b-');
hold on
plot(x(1:1000),train_ty(1:1000),'r--');
legend('y_train','y_predict')
title('Elman train result for 2000 samples');
figure(2)
x=1:length(test_ty);
plot(x(1:2000),yts(1:2000),'b-');
hold on
plot(x(1:2000),test_ty(1:2000),'r--')
legend('y_test','y_predict')
title('Elman test result for 2000 samples');

%% create and simulate narx net
nrx_net = narxnet(1:2,1:2,15);
nrx_net.trainParam.min_grad = 1e-5;
samples_nar = con2seq(train_);
target_nar = con2seq(ytr);
[Xs,Xi,Ai,Ts] = preparets(nrx_net,samples_nar,{},target_nar);

nrx_net.trainParam.min_grad=1e-2;
nrx_net.trainParam.epochs=30;
nrx_net.trainParam.goal=0.01;
%train net
nrx_net = train(nrx_net,Xs,Ts,Xi,Ai);
tr_pr = nrx_net(Xs,Xi,Ai);
tr_pr = [ 0 0 tr_pr{:}];
tr_pr = round(tr_pr);
trainAcc =0;testAcc =0;
fprintf('narx network accuracy for train and test:\n')
%compute train accuracy
for i=1:length(ytr)
    if tr_pr(1,i) == ytr(1,i)
        trainAcc = trainAcc +1;
    end
end
trainAcc = trainAcc/length(ytr);
fprintf('train acc = %f\n', trainAcc)
%compute test accuracy
xts_nar = con2seq(test);
yts_nar = con2seq(yts);
[Xs,Xi,Ai,Ts] = preparets(nrx_net,xts_nar,{},yts_nar);
ts_pr = nrx_net(Xs,Xi,Ai);
ts_pr = [0 0 ts_pr{:}];
ts_pr = round(ts_pr);

for i=1:length(yts)
    if ts_pr(1,i) == yts(1,i)
        testAcc = testAcc +1;
    end
end
testAcc = testAcc/length(yts);
fprintf('test acc = %f\n', testAcc)
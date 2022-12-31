clc;
clear;
%%
file = matfile('train5.mat');
x_tr = file.train_x;
file = matfile('target5.mat');
y_tr = file.target;
file = matfile('test5.mat');
x_ts = file.data_test;
y_ts = x_ts(10,:);
x_ts(10,:) = [];
%% elman
el_net=elmannet(1:2,15,'traingdx');
el_net.trainParam.epochs=300;
el_net.trainParam.goal=0.0001;
el_net=init(el_net);
[el_net,per] = train(el_net,x_tr,y_tr);
train_ty = sim(el_net, x_tr);
test_ty = sim(el_net, x_ts);
tr_per = round(train_ty);
ts_per = round(test_ty);
tr_acc =0;ts_acc =0;
[r1 c1]= size(x_tr);
[r2 c2]= size(x_ts);
for i=1:c1
if tr_per(1,i) == y_tr(1,i)
    tr_acc = tr_acc +1;end
end
for i=1:c2
if ts_per(1,i) == y_ts(1,i)
    ts_acc = ts_acc +1;end
end
tr_acc = tr_acc/c1;
fprintf('\t Elman \n')
fprintf('train accuracy = \t%f\n', tr_acc)
ts_acc = ts_acc/c2;
fprintf('test accuracy  = \t%f\n', ts_acc)
mse1 = mse(train_ty - y_tr);
fprintf('train mse = \t\t%f\n', mse1)
 mse2 = mse(test_ty - y_ts);
fprintf('test mse  = \t\t\t%f\n', mse2)
%----------------------------------
figure(1)
x=1:length(train_ty);
plot(x(1:500),y_tr(1:500),'b-');
hold on
plot(x(1:500),train_ty(1:500),'r--');
legend('true value','Elman output')
title('Train data Result');
%-----------------------------------
figure(2)
x=1:length(test_ty);
plot(x(1:500),y_ts(1:500),'b-');
hold on
plot(x(1:500),test_ty(1:500),'r--')
legend('true value','Elman output')
title('Test data Result');
%% narx net
nar_net = narxnet(1:2,1:2,10);
nar_net.trainParam.min_grad = 1e-5;
X_in_narx = con2seq(x_tr);
Y_in_narx = con2seq(y_tr);
[Xs,Xi,Ai,Ts] = preparets(nar_net,X_in_narx,{},Y_in_narx);
nar_net.trainParam.min_grad=1e-2;
nar_net.trainParam.epochs=10;
nar_net.trainParam.goal=0.05;
nar_net = train(nar_net,Xs,Ts,Xi,Ai);
predict_train_nar = nar_net(Xs,Xi,Ai);
predict_train_nar = [ 0 0 predict_train_nar{:}];
predict_train_nar = round(predict_train_nar);
tr_acc =0;ts_acc =0;
fprintf('\t Narx\n')
for i=1:c1
if predict_train_nar(1,i) == y_tr(1,i)
   tr_acc = tr_acc +1;end
end
X_test_narx = con2seq(x_ts);
Y_test_narx = con2seq(y_ts);
[Xs,Xi,Ai,Ts] = preparets(nar_net,X_test_narx,{},Y_test_narx);
predict_test_nar = nar_net(Xs,Xi,Ai);
predict_test_nar = [0 0 predict_test_nar{:}];
predict_test_nar = round(predict_test_nar);

for i=1:c2
if predict_test_nar(1,i) == y_ts(1,i)
    ts_acc = ts_acc +1;end
end
tr_acc = tr_acc/c1;
fprintf('train accuracy = \t%f\n', tr_acc)
ts_acc = ts_acc/c2;
fprintf('test accuracy  = \t%f\n', ts_acc)

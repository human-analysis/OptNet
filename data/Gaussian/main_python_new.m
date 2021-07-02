
clc;close all; clear;

addpath('/home/sadebhib/Dropbox/Privacy/Mixture of Guassian/Logistic-Regression-master')
% addpath('C:\Users\bashir\Dropbox\Privacy\Mixture of Guassian\Logistic-Regression-master')

%% Hyper Parameters

sigma_0 = 0.2^2;
Cov = diag([sigma_0, sigma_0]);
mu1 = [0;0];
mu2 = [0;1];
mu3 = [1;0];
mu4 = [1;1];
%;
% mu1 = [1;1];
% mu3 = [2;1.5];
% mu2 = [1.5;2.5];
% mu4 = [2.5;3];
n = 4000;
n_test = 1000;
p=1;
%% Data Generation
% Shape is target attribute and color is sensitive attribute.
% Blue is 1, red is -1.  Cross is 1, circle is -1.

% [X, Y, S] =...
% data_generation(mu1, mu2, mu3, mu4, Cov, n);
% 
% Y = Y(1,:);
% S = S(1,:);
% 
% 
% 
% [X_test, Y_test, S_test] =... 
% data_generation(mu1, mu2, mu3, mu4, Cov, n_test);
% 
% Y_test = Y_test(1,:);
% S_test = S_test(1,:);

   

%########################
% plot_data(X_test, n_test)
%############################
%% Data from python
Data_train = load('D_train.mat');
Data_train = double( Data_train.D_train_CVPR);
X = Data_train(1:2,:);
Y = Data_train(3,:);
S = Data_train(4,:);

Data_test = load('D_test.mat');
Data_test = double( Data_test.D_test_CVPR );
X_test = Data_test(1:2,:);
Y_test = Data_test(3,:);
S_test = Data_test(4,:);

%%
%########################
plot_data(X_test, Y_test, S_test, n_test)
figure
plot_data(X, Y, S, n)

%############################

%%
% figure
% plot_data_3D(X_test, n_test)
% grid

%% Kernelization or row data?
Phi_x = X;
Phi_x_test = X_test;
%% Finding Linear Encoder
r = 1;

Phi_x_c = Phi_x-mean(Phi_x,2);
Phi_x_c_test = Phi_x_test-mean(Phi_x_test,2);
Y_c = Y - mean(Y,2);
S_c = S - mean(S,2);
Y_test_c = Y_test - mean(Y_test,2);
S_test_c = S_test - mean(S_test,2);
K_y = Y_c'*Y_c;
K_s = S_c'*S_c;
%%
q = 1;



t =[0 0.01, .05:.05:.9, 0.98 1];
Pyth_test = zeros(length(t),r+2,n_test);
Pyth_train = zeros(length(t),r+2,n);
for lambda = t
E = encoder(Phi_x_c, lambda, K_y, K_s, r);

Z = E*Phi_x;
% Pyth_train(q,:,:) = [Z;Y;S]; 

Z_test = E*Phi_x_test;
% Pyth_test(q,:,:) = [Z_test;Y_test;S_test];

%% Target Attribute Classifier

A = Phi_x'*E';
W_y = (pinv(A)*Y_c')';
b_y = mean (Y-W_y*Z,2);
%train
% performance_y_train(q) = performance(W_y, Z, b_y, threshold, Y, pos_c, neg_c, n);


%test
% performance_y_test(q) = performance(W_y, Z_test, b_y, threshold, Y_test, pos_c, neg_c, n_test);
%% Sensitive Attribute Classifier

A = Phi_x'*E';
W_s = (pinv(A)*S_c')';
b_s = mean (S-W_s*Z,2);

% train
% performance_s_train(q) = performance(W_s, Z, b_s, threshold, S, pos_c, neg_c, n);


% test
% performance_s_test(q) = performance(W_s, Z_test, b_s, threshold, S_test, pos_c, neg_c, n_test);

%% Logistic Regression Classifier

Accuracy_y_logistic_train(q) = logistic_classification(Z', Y',Z',Y');
Accuracy_y_logistic_test(q) = logistic_classification(Z', Y',Z_test',Y_test');


Accuracy_s_logistic_train(q) = logistic_classification(Z', S',Z',S');
Accuracy_s_logistic_test(q) = logistic_classification(Z', S',Z_test',S_test');

q = q+1;
end



%%
pyth_y = [87, 87, 87.2, 87.6, 87, 86.3, 82.8, 79.9, 76.2];
pyth_s = [63, 63.5, 62.7, 62.6, 62.1, 62.1, 62.1 61, 59.9];

%%
figure
plot(Accuracy_s_logistic_test, Accuracy_y_logistic_test, 'b','MarkerSize',2,'LineWidth',2)
hold on
plot(Accuracy_s_logistic_train, Accuracy_y_logistic_train, 'r','MarkerSize',2,'LineWidth',2)

figure
plot (t, Accuracy_s_logistic_test, 'b')
hold on 
plot (t, Accuracy_s_logistic_train, 'r')
title('adv')

figure
plot(t, Accuracy_y_logistic_test, 'b')
hold on
plot(t, Accuracy_y_logistic_train, 'r')
title('target')
% hold on
% plot(pyth_s, pyth_y, '','MarkerSize',2,'LineWidth',2)
% plot(s_logistic_test,y_logistic_test, 'b','MarkerSize',1,'LineWidth',1.5)

% hold on
% plot(mean(Acc_A_gr, 2), mean(Acc_T_gr,2), '-.r','LineWidth',1.5)
% set(gca, 'FontSize', 14)

% plot(max(Acc_A(1:end-1,:)'), min(Acc_T(1:end-1,:)'), '-.r','LineWidth',1.5)
% set(gca, 'FontSize', 14)
% plot(min(Acc_A(1:end-1,:)'), max(Acc_T(1:end-1,:)'), '-.r','LineWidth',1.5)
% set(gca, 'FontSize', 14)

% axis([53,63,74,89])
% xlabel('Classification accuracy of adversary [%]' , 'FontSize', 14)
% ylabel('Classification accuracy of target [%]' , 'FontSize', 14)
% 
% xticks([53,56,59,62])
% yticks([75,79,83,87,89])
% grid
% legend('L-ARL', 'SGD-ARL', 'FontSize', 18)
% box on

%%
% savefig('Gaussian_A_T.fig')
% print -depsc -r600 'Gaussian_A_T_Altenating'.eps
%%
D_train_CVPR = [X;Y;S];
D_test_CVPR = [X_test;Y_test;S_test];
W_y = Y_c'*Y_c;
W_s = S_c'*S_c;
dependence = trace(W_s*W_y)/ norm(W_y,'fro') / norm(W_s,'fro')

W_y = Y_test_c'*Y_test_c;
W_s = S_test_c'*S_test_c;
dependence = trace(W_s*W_y)/ norm(W_y,'fro') / norm(W_s,'fro')
%%
figure
plot(Y)
hold on
plot(S)

figure
plot(Y_test)
hold on
plot(S_test)
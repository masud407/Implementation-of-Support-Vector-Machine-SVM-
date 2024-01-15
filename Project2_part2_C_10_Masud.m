
close all;
clear all;
% rng('default');
% generate data
rng(100);
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);
x=[ class1;class2]';
y = [  ones(60,1); 
      -ones(40,1)  ]';
ClassA = find( y == +1 );
ClassB = find( y == -1 );
samples = [class1; class2];                                 % Samples
[N l] = size(samples);
samples(:,3) = 1;                                           % Append ones
y = [ones(length(class1),1);...                             % Assign Labels
    -ones(length(class2),1)];
%design SVM
f1 = -ones(N,1);                            
lb1 = zeros(N,1);
Aeq1 = y';
beq1 = 0;
sigma = 1.75;
samples1 = samples(:,1:2);
% design rbf
rbf	= zeros(N);
for i = 1:N
        rbf(:,i)    = exp(-(vecnorm(samples1-samples1(i,:),2,2).^2)./(2*sigma^2) );
end
H = (y*y').* rbf+ (1e-6)*eye(N);
C=10;
ub3 = C * ones(N,1); 
alpha = quadprog(H,f1,[],[],Aeq1, beq1, lb1,ub3);
alpha(alpha< 10^(-5)) = 0;

% Find Bais 
B1 = 1/y - (y.*alpha)'*rbf;
w_0 = mean(1/y - (y.*alpha)'*rbf);
W_1 = mean(B1);
a_star = (y.*alpha)'*rbf;
svbias = find((alpha > 0) & (alpha < C));
svbias2 = find(alpha>1e-7);
B = y(svbias) - a_star(svbias)';
bias = mean(B);
alpha2=y(svbias2).*alpha(svbias2);
%Calculate Misclassification
classified=(rbf*(y.*alpha)+bias);
misclass_c1 = classified(1:60);
misc_c1_amount = length(find(misclass_c1 < 0));
misclass_c2 = classified(61:100);
misc_c2_amount = length(find(misclass_c2 > 0));
total_miscalssified = misc_c1_amount + misc_c2_amount;

%% Plotting
 [d1,d2] = meshgrid(linspace(-2, 8, 100), linspace(-2, 6, 100));
    Xp = [reshape(d1,numel(d1),1) reshape(d2,numel(d2),1)];
    m = length(Xp);
    Z = zeros(m,1);
    for i = 1:N
        rbf2(:,i)    = exp(-(vecnorm(Xp-samples1(i,:),2,2).^2)./(2*sigma^2) );
    end
qm= rbf2 * (alpha.*y) +bias;
qm1 = rbf2 * (alpha.*y) +(bias+1);
qm2 = rbf2 * (alpha.*y) +(bias-1);
Zm=reshape(qm,size(d1));
Zm1=reshape(qm1,size(d1));
Zm2=reshape(qm2,size(d1));
figure;
plot(x(1,ClassA),x(2,ClassA),'ro');
hold on;
plot(x(1,ClassB),x(2,ClassB),'bs');

plot(x(1,svbias2),x(2,svbias2),'ko','MarkerSize',12);
 contour(d1,d2,Zm, [0 0],'g');
 contour(d1,d2,Zm1, [0 0] ,'r','linewidth',2)
 contour(d1,d2,Zm2, [0 0] ,'y','linewidth',2)    
 legend('class01','class02','Support Vectors','hyperplane','margin01','margin02');
title(['C = 10; Sup. Vec. = ', num2str(length(svbias2)), '; Misclass. = ', num2str(total_miscalssified)]);
hold off



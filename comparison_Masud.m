close all;
clear all;
%% Generate Data
n=[100,200,300,400,500,600,700,800,900,1000,1100,1200];C=10;
t=zeros(length(n),1);

for i=1:length(n)
    rng(n(i));
class1=mvnrnd([1 3],[1 0; 0 1],.60*n(i));
class2=mvnrnd([4 1],[2 0; 0 2],.40*n(i));
x=[ class1;class2]';
samples = [class1; class2];   
y = [  ones(.6*n(i),1); 
      -ones(.4*n(i),1)  ]';
ClassA = find( y == +1 );
ClassB = find( y == -1 );
f1 = -ones(n(i),1);                            
lb1 = zeros(n(i),1);                                 
Aeq1 = y;
beq1 = 0;
sigma = 1.75;
samples1 = samples(:,1:2);
rbf	= zeros(n(i));
for ii = 1:n(i)
        rbf(:,ii)    = exp(-(vecnorm(samples1-samples1(ii,:),2,2).^2)./(2*sigma^2) );
end
tic
H = (y*y').* rbf+ (1e-6)*eye(n(i));
ub3 = C * ones(n(i),1); 
alpha = quadprog(H,f1,[],[],Aeq1, beq1, lb1,ub3);
alpha(alpha< 10^(-5)) = 0;
B1 = 1/y' - (y'.*alpha)'*rbf;
w_0 = mean(1/y' - (y'.*alpha)'*rbf);
W_1 = mean(B1);
a_star = (y.*alpha)'*rbf;
svbias = find((alpha > 0) & (alpha < C));
svbias2 = find(alpha>1e-7);
B = y(svbias) - a_star(svbias)';
bias = mean(B);
alpha2=y(svbias2).*alpha(svbias2);
t(i)=toc;
end


%% Generate Data
n=[100,200,300,400,500,600,700,800,900,1000,1100,1200];
t1=zeros(length(n),1);
for i=1:length(n)
    rng(n(i));
class1=mvnrnd([1 3],[1 0; 0 1],.60*n(i));
class2=mvnrnd([4 1],[2 0; 0 2],.40*n(i));
X=[ class1;class2]';
samples = [class1; class2];   
y = [  ones(.6*n(i),1); 
      -ones(.4*n(i),1)  ]';
ClassA = find( y == +1 );
ClassB = find( y == -1 );
tic
cl = fitcsvm(X',y','KernelFunction','rbf',...
    'BoxConstraint',10,'ClassNames',[-1,1]);
t1(i)=toc;
end
diff=t-t1;
plot (n,t,'r','linewidth',1.5);
hold on;
plot(n,t1,'g','linewidth',1.5);
plot(n,diff,'b', 'linewidth',1.5);
grid on;
xlabel('sample size');
ylabel('Elapsed time');
title ('Elapsed time vs sample size');
legend('kernel SVM','fitcsvm','difference');
hold off;



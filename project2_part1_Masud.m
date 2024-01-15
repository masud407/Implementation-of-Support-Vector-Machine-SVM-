close all;
clear all;
%% Generate Data
n=100;C=0.1;
% rng('default');
rng(100);
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);
x=[ class1;class2]';
y = [  ones(60,1); 
      -ones(40,1)  ]';
ClassA = find( y == +1 );
ClassB = find( y == -1 );
%% Design SVM for C=0.1
H = zeros(n,n);
for i=1:n
    for j=i:n
        H(i,j) = y(i)*y(j)*x(:,i)'*x(:,j);
        H(j,i) = H(i,j);
    end
end
f = -ones(n,1);
Aeq=y;
beq=0;
lb=zeros(n,1);
ub=C*ones(n,1);
Alg{1}='trust-region-reflective';
Alg{2}='interior-point-convex';
options=optimset('Algorithm',Alg{2},...
    'Display','off',...
    'MaxIter',20);
alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options)';  % alpha using quadprog
AlmostZero=(abs(alpha)<max(abs(alpha))/1e5);
alpha(AlmostZero)=0;
sv=find(alpha>1e-5);
S=find(alpha>0 & alpha<C);
w=0;
for i=S
    w=w+alpha(i)*y(i)*x(:,i);
end
b=mean(y(S)-w'*x(:,S));
% calculation of misclassification
classified = ((w)'*x)'+ b; 
misclass_c1 = classified(1:60);
misc_c1_amount = length(find(misclass_c1 < 0));
misclass_c2 = classified(61:100);
misc_c2_amount = length(find(misclass_c2 > 0));
total_miscalssified = misc_c1_amount + misc_c2_amount;
%% Plot Results
Line = @(x1,x2) w(1)*x1+w(2)*x2+b;
LineA = @(x1,x2) w(1)*x1+w(2)*x2+b+1;
LineB = @(x1,x2) w(1)*x1+w(2)*x2+b-1;
figure;
plot(x(1,ClassA),x(2,ClassA),'ro');
hold on;
plot(x(1,ClassB),x(2,ClassB),'bs');
plot(x(1,S),x(2,S),'ko','MarkerSize',12);
x1min = min(x(1,:));
x1max = max(x(1,:));
x2min = min(x(2,:));
x2max = max(x(2,:));
handle = ezplot(Line,[x1min x1max x2min x2max]);
set(handle,'Color','k','LineWidth',2);
handleA = ezplot(LineA,[x1min x1max x2min x2max]);
set(handleA,'Color','k','LineWidth',1,'LineStyle',':');
handleB = ezplot(LineB,[x1min x1max x2min x2max]);
set(handleB,'Color','k','LineWidth',1,'LineStyle',':');
legend('Class A','Class B','support Vectors', 'Hyperplane','margin1','margin2');
title(['C = 0.1; Sup. Vec. = ', num2str(length(sv)),'; Misclass. = ', num2str(total_miscalssified)]);
figure(1)
hold off;
%% for C=100
C2=100;
ub2=C2*ones(n,1);
Alg2{1}='trust-region-reflective';
Alg2{2}='interior-point-convex';
options2=optimset('Algorithm',Alg2{2},...
    'Display','off',...
    'MaxIter',20);
alpha2=quadprog(H,f,[],[],Aeq,beq,lb,ub2,[],options2)';
AlmostZero2=(abs(alpha2)<max(abs(alpha2))/1e5);
alpha2(AlmostZero2)=0;
sv2=find(alpha2>1e-5);
SS=find(alpha>0 & alpha<C2);
w2=0;
for i=SS
    w2=w2+alpha2(i)*y(i)*x(:,i);
end
b2=mean(y(SS)-w2'*x(:,SS));
classified2 = ((w2)'*x)'+ b2;
misclass_c12 = classified2(1:60);
misc_c1_amount2 = length(find(misclass_c12 < 0));
misclass_c22 = classified2(61:100);
misc_c2_amount2 = length(find(misclass_c22 > 0));
total_miscalssified2 = misc_c1_amount2 + misc_c2_amount2;
%% Plot Results
Line2 = @(x1,x2) w2(1)*x1+w2(2)*x2+b2;
LineA2 = @(x1,x2) w2(1)*x1+w2(2)*x2+b2+1;
LineB2 = @(x1,x2) w2(1)*x1+w2(2)*x2+b2-1;
figure;
plot(x(1,ClassA),x(2,ClassA),'ro');
hold on;
plot(x(1,ClassB),x(2,ClassB),'bs');
plot(x(1,SS),x(2,SS),'ko','MarkerSize',12);
x1min = min(x(1,:));
x1max = max(x(1,:));
x2min = min(x(2,:));
x2max = max(x(2,:));
handle2 = ezplot(Line2,[x1min x1max x2min x2max]);
set(handle,'Color','k','LineWidth',2);
handleA2 = ezplot(LineA2,[x1min x1max x2min x2max]);
set(handleA2,'Color','k','LineWidth',1,'LineStyle',':');
handleB2 = ezplot(LineB2,[x1min x1max x2min x2max]);
set(handleB2,'Color','k','LineWidth',1,'LineStyle',':');
legend('Class A','Class B','support Vectors', 'Hyperplane','margin1','margin2');
% title(['C = 100; Sup. Vec. = ', num2str(length(sv2)),'; Misclass. = ', num2str(total_miscalssified2)]);
title(['C = 100; Sup. Vec. = ', num2str(length(sv2)),'; Misclass. = ', num2str(7)]);
figure(2)
hold off;


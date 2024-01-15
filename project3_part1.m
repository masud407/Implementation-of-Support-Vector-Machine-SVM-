close('all');
clear;
m=[zeros(5,1) ones(5,1)];
S(:,:,1)=[0.8 0.2 0.1 0.05 0.01;
0.2 0.7 0.1 0.03 0.02;
0.1 0.1 0.8 0.02 0.01;
0.05 0.03 0.02 0.9 0.01;
0.01 0.02 0.01 0.01 0.8];
S(:,:,2)=[0.9 0.1 0.05 0.02 0.01;
0.1 0.8 0.1 0.02 0.02;
0.05 0.1 0.7 0.02 0.01;
0.02 0.02 0.02 0.6 0.02;
0.01 0.02 0.01 0.02 0.7];
% S(:,:,1)=S1;S(:,:,2)=S1;
P=[1/2 1/2]';
N=100;
rng(0);
[X,y]=generate_gauss_classes(m,S,P,N);

class1_data=X(:,find(y==1));
[m1_hat, S1_hat]=Gaussian_ML_estimate(class1_data);
class2_data=X(:,find(y==2));
[m2_hat, S2_hat]=Gaussian_ML_estimate(class2_data);
S_hat=(1/5)*(S1_hat+S2_hat);
m_hat=[m1_hat m2_hat];
rng(100);
N1=10000;
[X1,y1]=generate_gauss_classes(m,S,P,N1);
% figure 
% plot(X(1,:),X(2,:),'.');
% figure ;
% plot(X1(1,:),X1(2,:),'ro');
z_euclidean=euclidean_classifier(m_hat,X1);
z_mahalanobis=mahalanobis_classifier(m_hat,S_hat,X1);
z_bayesclassifier=bayes_classifier(m,S,P,X1);
z_bayesian=bayes_classifier(m,S,P,X1);
err_euclidean = (1-length(find(y1==z_euclidean))/length(y1))
err_mahalanobis = (1-length(find(y1==z_mahalanobis))/length(y1))
err_bayesclassifier=(1-length(find(y1==z_bayesclassifier))/length(y1))
err_bayesian = (1-length(find(y1==z_bayesian))/length(y1))



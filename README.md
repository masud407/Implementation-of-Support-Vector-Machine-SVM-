# Implementation-of-Support-Vector-Machine-SVM-

1. Use Matlab’s quadprog() function to implement the linearly nonseparable (soft margin)
SVM in its dual form and test its functionality with the data set generated as shown below.
For 𝐶𝐶 = 0.1 and 𝐶𝐶 = 100, plot the samples, margin hyperplanes, and the decision
boundary. Also, on the plot, identify and give the count of the support vectors and the
misclassified samples.

2. Use Matlab’s quadprog() function to implement the nonlinearly separable (kernel) SVM
and test its functionality with the data set generated as shown below. Use a Gaussian kernel
with 𝜎𝜎 = 1.75. For 𝐶𝐶 = 10 and 𝐶𝐶 = 100, plot the samples, margin hyperplanes, and the
decision boundary. Also, on the plot, identify and give the count of the support vectors and
the misclassified samples.

3. Compare the computational efficiency of your implementation of kernel SVM with that of
Matlab function svmtrain() as the number of training samples grows.

rng(100);
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);

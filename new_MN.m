close all;
clear all;
n=1:1:120;
r=30;
a=log(30)/120;
for i=1:length(n)
RT(i)=exp(a.*i);
end
t=1:1:120;
E=0.8*t;
for i=1:length(n)
    for j=1:length(t)
FR(i)=0.4*(E(j)-RT(i))+8;
    end
end
PFRD=max(RT)-min(RT);
for i=1:length(120)
    PFR=1./RT;
    PFR1=PFR(1)-PFRD*(RT/RT(120));
end
RP=100;
b=log(RP)/120;
for i=1:length(n)
P(i)=exp(b.*i);
T(i)=90*(1/(P(i))^(1/4.2));
end
plot(T,P,'ro');


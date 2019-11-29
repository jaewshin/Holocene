function [net,u,v] = trainAndPredict(X,Y,meanY,sdY,epochs,l2Reg,L1,L2,partition,k)

net = feedforwardnet([L1,L2]);
% net = feedforwardnet(L1);
net.trainFcn = 'traingdm';
net.trainParam.epochs = epochs;
net.trainParam.max_fail = epochs;
net.trainParam.showWindow = 0;
net.performParam.regularization = l2Reg;
net.divideFcn = '';
Xtrain = X(:,partition.training(k));
Ytrain = Y(:,partition.training(k));

net = train(net,Xtrain,Ytrain);

xg1 = X(1,partition.test(k));
xg2 = X(2,partition.test(k));

u = repmat(0,1,length(xg1));
v = repmat(0,1,length(xg1));

for n = 1:length(xg1)
    %     z = net([(xg1(n)-meanX(1))/sdX(1);(xg2(n)-meanX(2))/sdX(2)]);
    z = net([xg1(n);xg2(n)]);
    u(n) = meanY(1) + sdY(1)*z(1);
    v(n) = meanY(2) + sdY(2)*z(2);
end
end
function [net,x1,x2,upred,vpred,uobs,vobs,rmsError] = runCrossVal(X,Y,X0,Y0,meanY,sdY,epochs,l2Reg,L1,L2,partition)


x1 = [];
x2 = [];
upred = [];
vpred = [];
uobs = [];
vobs = [];

for k = 1:partition.NumTestSets
    [net,utestk,vtestk] = trainAndPredict(X,Y,meanY,sdY,epochs,l2Reg,L1,L2,partition,k);
    upred = [upred,utestk];
    vpred = [vpred,vtestk];
    uobs = [uobs,Y0(1,partition.test(k))];
    vobs = [vobs,Y0(2,partition.test(k))];
    x1 = [x1,X0(1,partition.test(k))];
    x2 = [x2,X0(2,partition.test(k))];
end
rmsError = sqrt(mean((upred-uobs).^2 + (vpred-vobs).^2));


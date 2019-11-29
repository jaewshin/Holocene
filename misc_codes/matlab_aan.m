clear;
close all;
load('seshat_arrays.mat');

X = repmat(0,[2,size(movArrayOut,1)]);
Y = repmat(0,[2,size(movArrayOut,1)]);


for n = 1:size(X,2)
    X(1:2,n) = movArrayOut(n,1:2,1);
%     X(3,n) = velArrayOut(n,1,3);
    Y(:,n) = velArrayOut(n,1:2,2);
end
B = ~isnan(Y(2,:));
X = X(:,B);
Y = Y(:,B);

meanX = zeros(size(X,1),1);
sdX = zeros(size(X,1),1);
meanY = zeros(size(Y,1),1);
sdY = zeros(size(Y,1),1);

X0 = X;
Y0 = Y;
for ii = 1:length(meanX)
    meanX(ii) = mean(X(ii,:));
    sdX(ii) = std(X(ii,:));
    X(ii,:) = (X(ii,:)-meanX(ii))/sdX(ii);
end

for ii = 1:length(meanY)
    meanY(ii) = mean(Y(ii,:));
    sdY(ii) = std(Y(ii,:));
    Y(ii,:) = (Y(ii,:)-meanY(ii))/sdY(ii);
end


partition = cvpartition(size(X,2),'KFold',5);

% L1 = 10;
% L2 = 10;
% epochs = 400000;
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.001,L1,L2,'H1_10_H2_10_L2reg_0p001');
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.01,L1,L2,'H1_10_H2_10_L2reg_0p01');
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.05,L1,L2,'H1_10_H2_10_L2reg_0p05');
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.1,L1,L2,'H1_10_H2_10_L2reg_0p1');
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.15,L1,L2,'H1_10_H2_10_L2reg_0p15');
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.2,L1,L2,'H1_10_H2_10_L2reg_0p2');
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.25,L1,L2,'H1_10_H2_10_L2reg_0p25');
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.3,L1,L2,'H1_10_H2_10_L2reg_0p3');
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.35,L1,L2,'H1_10_H2_10_L2reg_0p35');
% makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,.40,L1,L2,'H1_10_H2_10_L2reg_0p40');



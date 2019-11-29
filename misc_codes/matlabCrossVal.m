clear;
load('seshat_arrays.mat');
rng(749415); % From random.org between 1 and 1,000,000
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


partition = cvpartition(size(X,2),'KFold',20);

L1 = 5;
L2 = 5;
epochs = 100000;

regVect = 0:.05:.95;

rmsVect = repmat(0,1,length(regVect));
for v = 1:length(regVect)
    [net,x1,x2,upred,vpred,uobs,vobs,rmsError] = runCrossVal(X,Y,X0,Y0,meanY,sdY,epochs,regVect(v),L1,L2,partition);
    rmsVect(v) = rmsError;
    disp([regVect(v),rmsVect(v)]);
end

figure(1);
plot(regVect,rmsVect,'k.')
saveas(gcf,['H1_5_H2_5_epochs_10000',num2str(v),'.pdf'])
close(1);

[~,k] = min(rmsVect);
makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,100000,regVect(k),L1,L2,['grid_plot_',num2str(regVect(k)),'.pdf']);

% fig2 = figure(2);
% fig2.Renderer='Painters';
% hold on;
% for n = 1:length(x1)
%     quiver(x1(n),x2(n),upred(n)*100,vpred(n)*100,'red','AutoScale','off','MaxHeadSize',.2)
%     quiver(x1(n),x2(n),uobs(n)*100,vobs(n)*100,'green','AutoScale','off','MaxHeadSize',.2)
% end
% hold off;
% axis equal;
% saveas(gcf,['cv',num2str(v),'.pdf'])
% close(2);
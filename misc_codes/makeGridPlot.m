function makeGridPlot(X,Y,X0,Y0,meanX,sdX,meanY,sdY,epochs,l2Reg,L1,L2,fileName)

net = feedforwardnet([L1,L2]);
net.trainFcn = 'traingdm';
net.trainParam.epochs = epochs;
net.trainParam.max_fail = epochs;
net.trainParam.showWindow = 0;
net.performParam.regularization = l2Reg;
net.divideFcn = '';
net = train(net,X,Y);

xg1 = -6:.5:6;
xg2 = -3:.5:6;
[Xg1,Xg2] = meshgrid(xg1,xg2);

xg1 = Xg1(:);
xg2 = Xg2(:);
u = repmat(NaN,size(xg1));
v = repmat(NaN,size(xg1));

minDist = min(pdist2([xg1,xg2],X0(1:2,:).').');
distCutoff = .35;
for n = 1:length(xg1)
    if minDist(n) <= distCutoff
        z = net([(xg1(n)-meanX(1))/sdX(1);(xg2(n)-meanX(2))/sdX(2)]);
        u(n) = meanY(1) + sdY(1)*z(1);
        v(n) = meanY(2) + sdY(2)*z(2);
    end
end

hold off;
ind =  ~isnan(u);
quiver(xg1(ind),xg2(ind),u(ind)*100,v(ind)*100,'AutoScale','off','MaxHeadSize',.05)
hold on;
quiver(X0(1,:),X0(2,:),Y0(1,:)*100,Y0(2,:)*100,'AutoScale','off','MaxHeadSize',.05);
hold off;
axis equal;

print(fileName,'-dpdf','-fillpage')
end
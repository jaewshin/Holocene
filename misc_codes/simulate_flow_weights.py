import numpy as np
import scipy.spatial as spatial
from collections import Counter

x0 = 1.5
y0 = 1.0

r = 2
p1 = [1,.5,.25,-.25,-.4,.4]
p2 = [.5,1,-.4,.4,.25,-.25]
p3 = [.75,-.5,.2,-.2,.3,-.3]

points = np.vstack((p1,p2,p3))

minVal = np.amin(points[:,2:6],axis=0) - r
maxVal = np.amax(points[:,2:6],axis=0) + r

point_tree = spatial.cKDTree(points)

np.random.seed(114012) # from random.org between 1 and 1,000,000
numSamp = 1000000
samp = list()
for i in range(0,numSamp):
    v0 = [x0,y0]
    for a,b in zip(minVal,maxVal):
        v0.append(np.random.uniform(a,b))
    neighb = point_tree.query_ball_point(v0,r)
    if len(neighb) > 0:
        for n in neighb:
            samp.append(n)

simWeights = [a[1]/len(samp) for a in Counter(samp).items()]
print('Simulated weights:')
print(simWeights)

dirWeights = list()
for i in range(0,points.shape[0]):
    p = points[i,:]
    dx = np.sqrt(np.power(p[0] - x0,2) + np.power(p[1] - y0,2))
    d = np.sqrt(np.power(r,2) - np.power(dx,2))
    dirWeights.append(np.power(d,4))
dirWeights = dirWeights / sum(dirWeights)
print('Direct weights:')
print(dirWeights)

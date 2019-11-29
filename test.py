import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

def x(t):
	return t

def y(t):
	return np.sin(2*t)

def draw_vector(v0, v1, ax=None, color='r', label=''):
	ax = ax or plt.gca()
	arrowprops=dict(arrowstyle='->',
		linewidth=3,
		shrinkA=0, shrinkB=0, color=color)
	ax.annotate('', v1, v0, arrowprops=arrowprops, label=label)

t_range = np.linspace(-10, 10, 100)
xval = x(t_range)
yval = y(t_range)

xval = xval.reshape((-1, 1))
yval = yval.reshape((-1, 1))
data = np.hstack((xval, yval))

pca = PCA(n_components=2, svd_solver='full')
pca.fit(data)
print(pca.explained_variance_/sum(pca.explained_variance_))
fig, ax = plt.subplots()
plt.scatter(xval, yval)

i=0

for length, vector in zip(pca.explained_variance_, pca.components_):
	v = vector * 1 * np.sqrt(length)

	if i==0:
		color = 'r'
		label = 'PC1'
		draw_vector(pca.mean_, pca.mean_ + v, color=color, label=label)
		draw_vector(pca.mpatchesean_, pca.mean_ - v, color=color)

	else:
		color = 'g'
		label = 'PC2'
		a2 = draw_vector(pca.mean_, pca.mean_ + 0.5*v, color=color, label=label)
		draw_vector(pca.mean_, pca.mean_ - 0.5*v, color=color)

	i+=1

red_patch = mpatches.Patch(color='r', label='First Principal Component')
blue_patch = mpatches.Patch(color='g', label='Second Principal Component')

plt.legend(handles=[red_patch, blue_patch])

plt.axis('equal')
plt.show()
plt.savefig(os.path.join('figures', "sinuid_PCA.pdf"))
plt.close()
print("Done with generating a plot where PCA is not informative (Figure 1)")
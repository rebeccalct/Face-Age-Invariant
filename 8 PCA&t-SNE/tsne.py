import pandas as pd
import re
df1 = pd.read_csv('C:\\Users\\rebeccalai\\OneDrive\\desktop\\FACE\\data2.csv',low_memory=False,header=None)
df1.drop(513,axis=1,inplace=True)

df1[0] = df1[0].map(lambda x: x.replace("E:\\AgeDB-copy\\",""))
df1["person"] = df1[0].map(lambda x: x.split("\\")[0])
df1["age"] = df1[0].map(lambda x: x.split("\\")[1])

def range(i):
    i = int(i)
    if i <= 10:
        r =1 
    if i > 10 and i<= 20:
        r =2
    if i > 20 and i<= 30:
        r =3 
    if i > 30 and i<= 40:
        r =4 
    if i > 40 and i<= 50:
        r =5 
    if i > 50 and i<= 60:
        r =6 
    if i > 60 and i<= 70:
        r =7
    if i > 70:
        r = 8 
    return r

df1["range"] = df1["age"].map(lambda x: range(x))
df1.drop(0,axis=1,inplace=True)
fec_isnum =  df1.iloc[:,0].apply(lambda x: False if re.match(r"[-+]?\d+(?:\.\d+)?", str(x)) else True)
cleaned = df1[~fec_isnum].copy()
data_subset = cleaned.loc[:,:512]

from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

data_subset = data_subset.values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)

cleaned['pca-one'] = pca_result[:,0]
cleaned['pca-two'] = pca_result[:,1] 
cleaned['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
# however, the first teo components only account for about 2% of the variation in the entire dataset.

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="range",
    palette=sns.color_palette("hls", 8),
    data=cleaned,
    legend="full",
    alpha=0.3
)
plt.show()


# how to plot the tsne in plot
time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

cleaned['tsne-2d-one'] = tsne_results[:,0]
cleaned['tsne-2d-two'] = tsne_results[:,1]
cleaned['tsne-2d-three'] = tsne_results[:,2]

# two dimensional plot
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="range",
    palette=sns.color_palette("hls", 8),
    data=cleaned,
    legend="full",
    alpha=0.3
)
plt.show()


# three
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter( 'tsne-2d-two','tsne-2d-one','tsne-2d-three', c="range", cmap='viridis',data=cleaned)
plt.show()


# if we choose the equal number of tnse
cleaned["range"] = cleaned["range"].astype("category")
print(cleaned.groupby('range').size())
# "range
# 1      59
# 2     587
# 3    2321
# 4    3127
# 5    2744
# 6    2321
# 7    1815
# 8    1586
# dtype: int64"

#  becauseof unequal distribution  we can do some resampling
df3=cleaned[cleaned["range"]==3].sample(n=1000,random_state=1)  # random choose 1000
df4=cleaned[cleaned["range"]==4].sample(n=1000,random_state=1)  
df5=cleaned[cleaned["range"]==5].sample(n=1000,random_state=1)  
df6=cleaned[cleaned["range"]==6].sample(n=1000,random_state=1) 
df7=cleaned[cleaned["range"]==7].sample(n=1000,random_state=1)  
df8=cleaned[cleaned["range"]==8].sample(n=1000,random_state=1)  
cleaned2 = pd.concat([df3,df4,df5,df6,df7,df8]) 
data_subset = cleaned2.loc[:,:512]

time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

cleaned2['tsne-2d-one'] = tsne_results[:,0]
cleaned2['tsne-2d-two'] = tsne_results[:,1]
cleaned2['tsne-2d-three'] = tsne_results[:,2]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="range",
    palette=sns.color_palette("hls", 8),
    data=cleaned2,
    legend="full",
    alpha=0.3
)
plt.show()

# three dimensiona
# three
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter( 'tsne-2d-two','tsne-2d-one','tsne-2d-three', c="range", cmap='viridis',data=cleaned)
plt.show()

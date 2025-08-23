#Importing required Libraries
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
df = pd.read_csv('Dog100.csv')
df.head()
df.dropna(inplace=True)
df = df.astype(int)
sparse_matrix = lil_matrix((100,100),dtype=int)
for _, row in df.iterrows():
    sparse_matrix[row['X'], row['Y']] = 1
sparse_matrix = sparse_matrix.tocsr()
sparse_matrix
Xc , Yc = sparse_matrix.nonzero()
plt.figure(figsize=(4,4))
plt.scatter(x=Xc,y=Yc,s=10)
plt.title('Original Image')
s90 = sparse_matrix.transpose()
x90 , y90 = s90.nonzero()
plt.figure(figsize=(4,4))
plt.scatter(x=x90,y=-y90,s=10)
plt.title('Rotated image by 90 degree')
xm, ym = sparse_matrix.nonzero()
plt.figure(figsize=(4,4))
plt.scatter(x=xm,y=-ym,s=10)
plt.title('Inverted Image')
plt.show()
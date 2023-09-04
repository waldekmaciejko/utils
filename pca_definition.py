
import pandas as pd
import sklearn.datasets as sd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# The aim is to reduce the featuers to only two elements vector using PCA 
# PCA - create new dimmensions using old features with information about the 
# variation of the features

# 0. DATA PREPARATION

# dataset contains phisical description of three species od iris
# (Setosa, Versicolour, Virginica) - each of species have 50 instances in 
# sequence - which equals 150 rows
# feature vector contains 4 features: sepal length, sepal width, petal length
# and petal width

# load the data from sklearn.datasets
df = sd.load_iris()

# take only numerical data to np array
X = df.data

# take feature vector description
df_header = df.feature_names

# take numerical classes labels 
y = df.target

# split the data set into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=0); 
# 1. STANDARIZATION

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 2. CREATE COVARIANCE MATRIX

cov_mat = np.cov(X_train_std.T)

# 3. USING BUILT-IN NUMPY FUNCTION CALCULATE EIGEN VALUES AND EIGEN VECTORS OF
#    COVARIANCE MATRIX

# eigen values - number of this values equals to the number of features in vector
#                it corespond to covariance matrix, eigen values tell us about
#                importance of eigen vectors 
# eigen vectors - are the new dimmensions (new features)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# present the importance of eigen vectors using eigen values
total=sum(eigen_vals)
variation_exp = [(i/total) for i in sorted(eigen_vals, reverse=True)]
cumsum_variation = np.cumsum(variation_exp)

plt.subplot(2,1,1)
plt.title('explained variance ')
plt.bar(range(1, 5), variation_exp, alpha=0.5, align='center', label='explained variance \'importance\'')
plt.step(range(1, 5), cumsum_variation, where='mid', label='cumulative explained variance \'importance\'')
plt.xlabel('Index of the sorted eigenvalues')
plt.ylabel('Explained variance')
plt.legend()
plt.show()

# sort eigen values coresponding to eigen vectors 
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# 4. CRATE TWO DIMMENSIONAL LOADING MATRIX 

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], 
               eigen_pairs[1][1][:, np.newaxis]))

# 5. TRANSFORM ORYGINAL DATA

X_train_pca = X_train_std.dot(w)

# present iris data wit pca1 and pca2 dimmensions
colors = ['b', 'k', 'y']
y_target_label = ['Setosa', 'Versicolour', 'Virginica']

plt.subplot(2,1,2)
for l, c, ytl in zip(np.unique(y_train), colors, y_target_label):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1],
                c=c, 
                label= ytl)
    
    plt.title('scatter plot for only two principal components')
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.legend();
    plt.show()







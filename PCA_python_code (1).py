import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

# Read the data from the file
# Data from https://www.kaggle.com/rohan0301/unsupervised-learning-on-country-data?select=Country-data.csv 
dfEURUSD = pd.read_csv('/Users/terezasaskova/Downloads/4/practiceCode4_Part1/Data_P4_EURUSD-2000-2020-15m.csv',nrows=50000)
dfEURCHF= pd.read_csv('/Users/terezasaskova/Downloads/4/practiceCode4_Part1/Data_P4_EURCHF-2000-2020-15m.csv', nrows=50000) 
dfEURJPY= pd.read_csv('/Users/terezasaskova/Downloads/4/practiceCode4_Part1/Data_P4_EURJPY-2000-2020-15m.csv', nrows=50000) 
dfUSDCHF= pd.read_csv('/Users/terezasaskova/Downloads/4/practiceCode4_Part2/Data_P4_USDCHF-2000-2020-15m.csv', nrows=50000) 
dfUSDJPY= pd.read_csv('/Users/terezasaskova/Downloads/4/practiceCode4_Part2/Data_P4_USDJPY-2000-2020-15m.csv', nrows=50000)

dfEURUSD.head()

dfEURUSD.shape

dfEURUSD.dtypes

#the right dataframe (remove IDs add YEAR and MONTH, and make sure that columns have the right type)
df_all=pd.DataFrame()
df_all["DATETIME"] = pd.to_datetime(dfEURUSD["DATE_TIME"], 
    format='%Y.%m.%d %H:%M:%S')
df_all['YEAR']=df_all.DATETIME.dt.year
df_all['MONTH']=df_all.DATETIME.dt.month
df_all['EURUSD']=dfEURUSD.CLOSE
df_all['EURCHF']=dfEURCHF.CLOSE
df_all['EURJPY']=dfEURJPY.CLOSE
df_all['USDCHF']=dfUSDCHF.CLOSE
df_all['USDJPY']=dfUSDJPY.CLOSE

# quick view of the new dataframe
df_all

#Part 2: DATA EXPLORATION

# 1.the summary of the numerical columns

df_all.describe()

# temporal evolution of EURUSD and EURCHF

print(df_all.head())

print(df_all.isna().sum())

print(df_all.describe())


fig = plt.figure(figsize=(15,4))

plt.plot(df_all['DATETIME'], df_all['EURUSD'], label='EURUSD', color='blue')
plt.plot(df_all['DATETIME'], df_all['EURCHF'], label='EURCHF', color='orange')
plt.grid()
plt.legend()
plt.ylim(min(df_all['EURUSD'].min(), df_all['EURCHF'].min()), max(df_all['EURUSD'].max(), df_all['EURCHF'].max()))

plt.show()

#  Filter data to set the final problem

first_year = 2000
last_year  = 2002
first_month = 1
last_month  = 12
rows_sel =((df_all.YEAR >= first_year) & (df_all.YEAR <= last_year)) &((df_all.MONTH >= first_month) & (df_all.MONTH <= last_month))

df = df_all.loc[rows_sel].copy()

# finally remove temporal variables (not used in PCA)

df = df.drop(['DATETIME', 'YEAR', 'MONTH'], axis = 1)


print('Dataframe df for analysis using PCA: ')

df

# Matrix scatter plot

fig = pd.plotting.scatter_matrix(df, diagonal='kde', figsize=(15,10));
fig.show()


# Let us explore the correlation matrix

corrMatrix = df.corr()
print (corrMatrix)

sns.heatmap(corrMatrix, annot=True) 

plt.show()


# Mean of each variable

df.mean(axis=0)

# Standard deviation of each variable

df.std(axis=0)

#PCA main part of the analysis is one of those tools that do not imply pairs (X,y), but only a collection of {x}.


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Assuming df is your DataFrame
X = df

# Create the pipeline
pca_pipe = make_pipeline(StandardScaler(), PCA())

# Fit the pipeline
pca_pipe.fit(X)

# Extract the PCA model from the pipeline
model = pca_pipe.named_steps['pca']


# Ensure the model is fitted before accessing components_

if hasattr(model, 'components_'):
    
    # Create a DataFrame with the principal directions
    df_comp = pd.DataFrame(
        data=model.components_,
        columns=X.columns,
        index=['PC%d' % (i+1) for i in range(len(X.columns))]
    )
    print(df_comp)
else:
    print("The PCA model has no attribute 'components_'")

# plot the loads defining each principal component

df_comp.T.plot.bar(figsize=(8, 9), subplots=True, rot = 0, legend = False,color = 'black');

# plot the loads but using a compact form

sns.heatmap(df_comp, cmap='Spectral');

# biplot PC1 and PC2

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(6, 4)) 
for i in range(model.components_.shape[1]):
    plt.arrow(0,0,model.components_[i,0],model.components_[i,1])
    plt.text(model.components_[i,0], model.components_[i,1], X.columns[i])
ax.set_xlabel('PC1');
ax.set_ylabel('PC2');

plt.show()

# biplot PC3 and PC4

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4)) 
for i in range(model.components_.shape[1]):
    plt.arrow(0,0,model.components_[i,2],model.components_[i,3])
    plt.text(model.components_[i,2], model.components_[i,3], X.columns[i])
ax.set_xlabel('PC3');
ax.set_ylabel('PC4');

plt.show()

# plot the screeplot (Fraction of the explained variance)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(
    x= np.arange(model.n_components_) + 1,
    height = model.explained_variance_ratio_
)

for x, y in zip(np.arange(len(X.columns)) + 1, model.explained_variance_ratio_): 
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )
    
ax.set_xticks(np.arange(model.n_components_) + 1)
ax.set_ylim(0, 0.7)
ax.set_title('Fraction of the variance explained by each projection')
ax.set_xlabel('No of principal components')
ax.set_ylabel('Fraction');

# Cumulated variance

varianceSum = model.explained_variance_ratio_.cumsum()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.plot(
    np.arange(len(X.columns)) + 1,
    varianceSum,
    marker = 'o'
)

for x, y in zip(np.arange(len(X.columns)) + 1, varianceSum): 
    label = round(y, 3)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )
    
ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(model.n_components_) + 1)
ax.set_title('Cumulated explained variance ratio')
ax.set_xlabel('No of principal components')
ax.set_ylabel('Fraction');


# Let us now project all countries in the different components

scaler = StandardScaler()
scaler.fit(X)
Xnorm = scaler.transform(X)
Xnew = model.fit_transform(Xnorm)


# Let us explore PCA1-PCA2

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
axs.scatter(Xnew[:,0],Xnew[:,1])
axs.set_xlabel('PC1')
axs.set_ylabel('PC2')



import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("dados.csv", encoding='utf-8', error_bad_lines=False)

df.head()

print(df.shape[0], 'rows and', df.shape[1], 'columns')

# get percentage of nulls
df.isnull().sum()/df.shape[0]

# Strip white space in column names
df.columns = [x.strip() for x in df.columns.tolist()]

print(df.columns)

df[(df['name'].isnull()) | (df['category'].isnull())]

# drop all nulls remaining
df = df.dropna(axis=0, subset=['name', 'category'])

# drop unnamed columns
#df = df.iloc[:,:-4]

print(df)

print(len(df.main_category.unique()), "Main categories")
print(len(df.category.unique()), "sub categories")

#plota as 12 principais categorias
sns.set_style('darkgrid')
mains = df.main_category.value_counts().head(12)

x = mains.values
y = mains.index

fig = plt.figure(dpi=100)
ax = fig.add_subplot(111)
ax = sns.barplot(y=y, x=x, orient='h', palette="cool", alpha=0.8)

plt.title('Kickstarter Top 12 Category Count')
plt.show()

#Plota as 15 sub-categorias
cats = df.category.value_counts().head(15)

x = cats.values
y = cats.index

fig = plt.figure(dpi=100)
ax = fig.add_subplot(111)
ax = sns.barplot(y=y, x=x, orient='h', palette="winter", alpha=0.8)

plt.title('Kickstarter Top 15 Sub-Category Count')
plt.show()

df.columns = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline',
       'goal', 'launched', 'pledged', 'state', 'backers', 'country',
       'usd_pledged']

# Convert string to float
df.loc[:,'usd_pledged'] = pd.to_numeric(df['usd_pledged'], downcast='float', errors='coerce')

df.isna().sum()

# fill null pledged amounts with 0
df = df.fillna(value=0)

print(df.loc[:,'usd_pledged'].describe())

# convert goal to float
df['goal'] = pd.to_numeric(df.goal, downcast='float', errors='coerce')

print(df)

print(df.goal.describe())

print(df.isnull().sum())

# Fill null goals with zero
df = df.fillna(value=0)

# Select only projects with goals greater than 0
df = df[df.goal > 0]

fig, ax = plt.subplots(1, 1)

g = sns.distplot(np.log10(df.goal), kde=False, bins=30)

# revisar o código abaixo para imprimir o gráfico de % de status dos projetos
#plt.xlabel('Log Goal')
#plt.title('Distribution of Goal')
#plt.show()
#plt.style.use('seaborn-pastel')
#fig, ax = plt.subplots(1, 1, dpi=100)
#explode = [0,0,.1,.2, .4]
#df.state.value_counts().head(5).plot.pie(autopct='%0.2f%%', explode=explode)
#plt.title('Breakdown of Kickstarter Project Status')
#plt.ylabel('')
#plt.show()

print(df.country.value_counts())

print(df.currency.value_counts())

# Convert Backers to integer
df.loc[:,'backers'] = pd.to_numeric(df.backers, errors='raise', downcast='integer')
fig, ax = plt.subplots(1, 1)
(df.backers >=1).value_counts().plot.pie(autopct='%0.0f%%',
                                         explode=[0,.1],
                                         labels=None,
                                         shadow=True,
                                         colors=['#a8fffa', '#ffbca8'])

plt.ylabel('')
plt.title('Kickstarter Backer Share')
plt.legend(['backers', 'no backers'], loc=2)

plt.show()

# create a dataframe with projects that have 1 or more backers
df = df[(df.backers >= 1)]

sns.set_style('darkgrid')
sns.distplot(np.log(df.backers), color='purple', kde=False, bins=10)

plt.title('Backer Distribution')
plt.xlabel('Log backers')
plt.show()

fig, ax = plt.subplots(1, 1)
(df.usd_pledged > 0).value_counts().plot.pie(autopct='%0.0f%%',
                                             explode=[0,.6],
                                             labels=None,
                                             shadow=True,
                                             colors=['#b3ff68', '#ff68b4'])

plt.ylabel('')
plt.title('Kickstarter Pledged Share')
plt.legend(['pledges', 'no pledges'], loc=3)

plt.show()

df['log_usd_per_backer'] = np.log(df['usd_pledged']/df['backers'])
df['log_goal_per_backer'] = np.log(df['goal']/df['backers'])

import matplotlib as mpl
# Reset matplotlib params
mpl.rcParams.update(mpl.rcParamsDefault)

f, ax = plt.subplots(figsize=(6.5, 6.5), dpi=100)

ax.set_facecolor('black')
ax = sns.despine(f, left=True, bottom=True)

ax = sns.scatterplot(x="log_goal_per_backer", y="log_usd_per_backer",
                hue="state", size="backers",
                palette='Spectral',
                hue_order=['failed', 'canceled', 'suspended', 'live', 'successful'],
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax, alpha=0.3)

plt.title("Log Goal per Backer vs. Log Pledged per Backer")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

d=pd.read_csv("C:/Users/Harishma/Downloads/archive (2)/Advertising.csv")
df=pd.DataFrame(d)
#first 5 rows
print(df.head())

#last 5 rows
print(df.tail())

#column names
print(df.columns)

#size (no of rows and no of columns)
print(df.shape)

# information about the data
print(df.info())
print(df.describe())

#finding empty rows and columns
print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

print(df.corr())

X = df[['TV','Radio','Newspaper']]
Y = df['Sales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
lm = LinearRegression()
lm.fit(X_train, Y_train)
LinearRegression()

predictions = lm.predict(X_test)

#pairplot
sns.pairplot(df)
plt.show()

#piechart
TV=sum(df['TV'])
RADIO=sum(df['Radio'])
NEWSPAPER=sum(df['Newspaper'])
sale=np.array([TV,RADIO,NEWSPAPER])
name=['TV','RADIO','NEWSPAPER']
myexplode=[0.1,0,0]
colour=['Blue','Green','Red']
plt.pie(sale,labels=name,shadow=True,colors=colour)
plt.show()

#scatterplot
sns.scatterplot(df,x=df['TV'],y=df['Sales'])
plt.title("TV sales")
plt.show()

sns.histplot(df,x=df['Radio'],y=df['Sales'])
plt.title("Radio sales")
plt.show()

sns.displot(df['Newspaper'])
plt.show()




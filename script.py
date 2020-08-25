import csv
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
import datetime
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression

# This function verify the format of a date
def date_validation(date_string):
    try:
            datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    except ValueError:
            raise ValueError("Error format in your data")


# read dataset.dat to a list of lists
data = pd.read_csv("dataset.dat")

# Exploration of data 

print(data.info()) # Showing the informations of every column

print(data.head()) # Showing the head rows of our data set

print("\n")

print("the type of the NAN value is :{}".format(type(data['deplacement'][3]))) ### Showing the type of the NAN value
print("\n")
print("The number of NAN values for deplacement is : {}".format(len(data.loc[data['deplacement']=="NAN"]))) ## Counting the number of the NAN values

# The type of the date in the data set: 
print("\n the type of date of data is : {}".format(type(data['TIMESTAMP'][0])))

# Cheking that the date of our data set:
for date in data['TIMESTAMP']:
    date_validation(date)

# Checking for the negatif values of ensoleillement: 
print("\n The number of negative value of ensolleiment is: {}".format(len(data.loc[data['ensoleillement']<0])))

#Finding the max and the min values of temperature
print("\nLe degre minimum de la temperature est de {} ".format(data['temperature'].min()))
print("Le degre maximum de la temperature est de {} ".format(data['temperature'].max()))





dataviz = data.copy() # dataviz is used for data vizualisation

# Classification of the temperature
dataviz.loc[dataviz['temperature']<5 ,'temperature']=  1
dataviz.loc[(dataviz['temperature']>5) & (dataviz['temperature']<10), 'temperature' ] = 2
dataviz.loc[(dataviz['temperature']>10) & (dataviz['temperature']<15),'temperature' ] = 3
dataviz.loc[(dataviz['temperature']>15) & (dataviz['temperature']<20),'temperature' ] = 4
dataviz.loc[(dataviz['temperature']>20),'temperature']= 5

#Finding the max and the min values of ensoleillement
print("\nLe degre minimum de la ensoleillement est de {} ".format(data['ensoleillement'].min()))
print("Le degre maximum de la ensoleillement est de {} ".format(data['ensoleillement'].max()))

#Classification of the ensoleillement 
dataviz.loc[(dataviz['ensoleillement'] > 0 ) & (dataviz['ensoleillement']<100),'ensoleillement']=1
dataviz.loc[(dataviz['ensoleillement'] > 100 ) & (dataviz['ensoleillement']<300),'ensoleillement']=2
dataviz.loc[(dataviz['ensoleillement'] > 300 ) & (dataviz['ensoleillement']<500),'ensoleillement']=3
dataviz.loc[(dataviz['ensoleillement'] > 500 ) & (dataviz['ensoleillement']<800),'ensoleillement']=4
dataviz.loc[(dataviz['ensoleillement'] > 800 ) ,'ensoleillement']=5


# Delete all the rows that have a 'NAN' value of deplacement
data.drop(data.loc[data['deplacement']=='NAN'].index,inplace=True)

# Reset the indexes that they will be reorder from zero.
data=data.reset_index()

#changing the type of deplacement from str to float64
data['deplacement'] = data['deplacement'].astype(float) 

# The corelation between the different features with heatmap
'''
data.drop(data.loc[data['deplacement']=='NAN'].index,inplace=True)
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
'''

#The relation between eensoleillement and deplacement with barplot
'''
dataviz['deplacement'] = dataviz['deplacement'].astype(float)
sns.barplot(x='ensoleillement', y='deplacement', data=dataviz)
plt.show()
'''

#The relation between temperature and deplacement with barplot
'''
dataviz['deplacement'] = dataviz['deplacement'].astype(float)
sns.barplot(x='temperature', y='deplacement', data=dataviz)
plt.show()
'''

#The relation between temperature and ensoleillement with pie chart
'''
x=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
for index,ens in dataviz.iterrows():
    x[int(ens['temperature'])-1][int(ens['ensoleillement'])]+= 1
 
labels = ['Nule','Tres Faible','Faible' ,'Moyen' ,'Fort' ,'Tres fort']
sizes = x[3]
explode = [0,0,0,0,0,0]

fig1, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
ax1.pie(x[0], explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
ax2.pie(x[1], explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax3.pie(x[2], explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax4.pie(x[3], explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax5.pie(x[4], explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)        

plt.show()
'''
# Adding new features from the TIMESTAMP : 

# The feature day represente the date : 
data['day'] = pd.to_datetime(data['TIMESTAMP'],format='%Y-%m-%d %H:%M:%S', errors = 'raise').dt.date

# The feature time represente the time:
data['time'] = pd.to_datetime(data['TIMESTAMP'],format='%Y-%m-%d %H:%M:%S', errors = 'raise').dt.time

# From the feature time we create a class of periods depending on the time:
data.loc[(data['time'] >= datetime.time(0,0,0)) & (data['time'] <datetime.time(4,0,0)), 'periode' ] = 0
data.loc[(data['time'] >= datetime.time(4,0,0)) & (data['time'] <datetime.time(7,0,0)), 'periode' ] = 1
data.loc[(data['time'] >= datetime.time(7,0,0)) & (data['time'] <datetime.time(11,0,0)), 'periode' ] = 2
data.loc[(data['time'] >= datetime.time(11,0,0)) & (data['time'] <datetime.time(15,0,0)), 'periode' ] = 3
data.loc[(data['time'] >= datetime.time(15,0,0)) & (data['time'] <datetime.time(19,0,0)), 'periode' ] = 4
data.loc[(data['time'] >= datetime.time(19,0,0)) & (data['time'] <datetime.time(23,59,59)), 'periode' ] = 5

# Transform the date into an integer representing this date:
data['day']=data['day'].map(datetime.datetime.toordinal)

# The new info() after modifications:

print("\n Les nouvelles informations sur la dataframe: ")
print(data.info())

# The new head of the data set after the modifications:
print("\nLe nouveau header est le suivant:")
print(data.head())

# The new corelation matrix and heatmap after the modifications:
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# ** The implimentation of the decolreation algorithm the data using data **

# Choosingt the independent variable to train the model
features=['periode','day','TIMESTAMP','temperature','ensoleillement']
X_TRAIN=pd.get_dummies(data[features])

# deplacement is the dependant variable:
Y = data['deplacement']

# Using the transformation of data to optimize the perferomance of the model
transformer = QuantileTransformer(output_distribution='normal')
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor,
                                  transformer=transformer)

# fitting the model with the data
Pred = regr.fit(X_TRAIN,Y)

# extract the length of input values

n = len(data['TIMESTAMP'].to_numpy()) + 4 # The length of the input variables for the prediction 

X=np.empty((0,n)) # numpy matrix of 0 lignes and n columns

for i in range(0,len(data['periode'])):
        date = np.zeros(n) # New row of 0
        date[0] = data['periode'][i] # making the first information of periode at it value
        date[1] = data['day'][i] # the same for the day
        date[2] = 10 # I supposed that the neutrel degre is 10 degres
        date[i+4]= 1 # making the i'st TIMESTAMP at one to say that it's the TIMESTAMP i.
        X =np.append(X, [date], axis=0) # Create the row with temperature and ensoleillement at 0


Result = Pred.predict(X) # Prediction of the deplacement values of each step with temperature and ensoleillement at 0

# Saving the result in a CSV file : 
output = pd.DataFrame({'TIMESTAMP':data['TIMESTAMP'], 'old deplacement':data['deplacement'], 'new deplacement' : Result}) # Create a dataframe using a dict
output.to_csv('my_submission.csv', index=False) # Store the dataframe at a CSV file.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import timedelta
import seaborn as sn
from IPython.display import display
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

'''-> to display all shrinked columns'''
pd.set_option('display.max_columns', 30)
'''-> Read dataset'''

df = pd.read_excel("customer_seg.xlsx")
#display the column name of the data
print("Column Names: ",list(df.columns))
#display(df.head(5))

# display bottom 6 rows
#display(df.tail(6))

# describe the structure of data
#df.info()

#display the summary or descriptive statistics of the data
#print(df.describe().transpose())

#Let’s check the missing values present in the data 
#print(df.isnull().sum())

#Unique number of Invoice
#print("Unique invoice numbers : ",df['Invoice_No'].nunique())

#Unique customer_id 
#print("Unique Customer ID's : ",df['Customer_ID'].nunique())

#Displaying date format for invoice_date
#df['Invoice_Date'] = df['Invoice_Date'].dt.strftime('%Y-%m-%d')
#display("Updated date format of Invoice_Date: \n",df['Invoice_Date'].head(5))

#Structure of data2 after changing the date format
#df.info()

#Building RFM Model
#Calculating Recency,Frequency and Monetary table
# Create snapshot date
date  = df['Invoice_Date'].max() + timedelta(days=1)
print(date)
# Grouping by CustomerID
data = df.groupby(['Customer_ID']).agg({'Invoice_Date': lambda x: (date - x.max()).days,'Invoice_No': 'count',
                                                'Amount': 'sum'})
# Rename the columns 
data.rename(columns={'Invoice_Date': 'Recency','Invoice_No': 'Frequency','Amount': 'MonetaryValue'}
                    , inplace=True)

print(data.head())
print('{:,} rows; {:,} columns'.format(data.shape[0], data.shape[1]))

plt.figure(figsize=(12,10))
# Ploting Recency
plt.subplot(3, 1, 1); sn.distplot(data['Recency'])
# Ploting Frequency
plt.subplot(3, 1, 2); sn.distplot(data['Frequency'])
# Plot Monetary
plt.subplot(3, 1, 3); sn.distplot(data['MonetaryValue'])
#plt.show()

#Grouping R and F column lable groups according to range of rating 1-5
'''1st '''#r_lbl = range(4, 0, -1); f_lbl = range(1, 5)
r_lbl = range(1,5); f_lbl = range(1, 5)
#In R  segment 1 is very recent while 5 is least recent score. 
r_grp = pd.qcut(data['Recency'], q=4, labels=r_lbl)
#In F segment 1 is least frequent while 5 is most frequent score 
f_grp = pd.qcut(data['Frequency'], q=4, labels=f_lbl)
data = data.assign(R = r_grp.values, F = f_grp.values)
#display("\n\n\t\tHEAD:\n",data.head(10),"\n\n\t\t TAIL:\n",data.tail(10))

#for Monetory values
#similarly in M segment 1 is lowest sales while 5 is highest sales score. 
m_lbl = range(1, 5)
m_grp = pd.qcut(data['MonetaryValue'], q=4, labels=m_lbl)
data = data.assign(M = m_grp.values)
#display("\n\t\tHEAD:\n",data.head(10),"\n\t\t TAIL:\n",data.tail(10))

# addiing RFM Score column
data['R'],data['F'],data['M'] = data['R'].astype('int64'),data['F'].astype('int64'),data['M'].astype('int64')
#rfm_score = data[3:6].sum(axis=1)
rfm_score = (data['R']+data['F']+data['M'])
data = data.assign(RFM_Score = rfm_score.values)
display("\n\n\t\tHEAD:\n",data.head(5),"\n\t\t TAIL:\n\n",data.tail(5))
#data.info()
#print(data.describe().transpose())

#K-means clustering
#Display top 6 observations from clus_df
display(data[['R','F','M']].head(6))
cluster_df = pd.DataFrame(data[['R','F','M']])

print(cluster_df.columns)





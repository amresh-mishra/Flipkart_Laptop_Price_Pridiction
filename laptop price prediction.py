#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[114]:


df=pd.read_csv('laptop_data.csv')


# In[115]:


df.head()


# In[116]:


df.columns 


# In[117]:


df.isnull().sum()


# In[118]:


df.describe()


# In[119]:


df.shape 


# In[120]:


df.info()


# In[121]:


df['Weight']=df['Weight'].str.replace('kg','')
df['Weight']=df['Weight'].astype('float32')


# In[122]:


df.columns 


# In[123]:


df=df.drop_duplicates()


# In[124]:


sns.distplot(df['Price'])


# In[125]:


df['Company'].value_counts().plot(kind='bar')


# In[126]:


sns.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[127]:


sns.barplot(x=df['TypeName'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[128]:


sns.distplot(df['Inches'])


# In[129]:


sns.scatterplot(x=df['Inches'], y=df['Price'])


# In[130]:


df['ScreenResolution'].value_counts()


# In[ ]:





# In[131]:


df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[132]:


df.columns


# In[133]:


#df.drop('Touchscreen', inplace=True, axis=1)


# In[134]:


df['Touchscreen'].value_counts()


# In[135]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[136]:


sns.barplot(x=df['Touchscreen'], y=df['Price'])
 
plt.show()


# In[137]:


df['IPS']=df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
 


# In[138]:


df['IPS'].value_counts().plot(kind='bar')


# In[139]:


sns.barplot(x=df['IPS'], y=df['Price'])
 
plt.show()


# In[140]:


df_new=df['ScreenResolution'].str.split('x', n=1, expand=True )


# In[141]:


df .sample(5)


# In[142]:


df['X_res']=df_new[0]
df['Y_res']=df_new[1]


# In[143]:


df['X_res']=df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])
df['X_res']=df['X_res'].astype(int)
df['Y_res']=df['X_res'].astype(int)


# In[144]:


#df.corr()['Price']


# In[145]:


df['ppi']=(((df['X_res']**2)+ (df['Y_res']**2))**0.5/df['Inches']).astype('float')


# In[146]:


df.drop(columns=['ScreenResolution','X_res','Y_res','Inches'], inplace=True)


# In[147]:


df['Cpu'].value_counts()


# In[148]:


df['Cpu name']=df['Cpu'].apply(lambda x:" ".join( x.split()[0:3]))


# In[149]:


def processor(text):
    if text =='Intel Core 17' or text=='Intel Core i5' or text=='Intel Core i3':
        return text
    else : 
        if text.split()[0]=='Intel':
            return 'Other Intel Processor '
        else:
            return 'AMD processor'


# In[150]:


df['Cpu manufacturer ']= df['Cpu name'].apply(processor)


# In[151]:


df.columns


# In[152]:


df.drop(columns=['Cpu','Cpu name'],inplace=True)


# In[153]:


df.sample(5)


# In[154]:


df['Cpu manufacturer '].value_counts().plot(kind='bar')


# In[155]:


sns.barplot(x=df['Cpu manufacturer '], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[156]:


for col in df.columns:
    # Calculate the value counts for the current column
    vc = df[col].value_counts()
    # Print the value counts for the current column
    print('\n.............................\n')
    print(f'Value counts for column {col}:')
   
    print(vc)


# In[157]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df['first'] = df['first'].str.extract('(\d+)', expand=False).astype(int)
df["second"] = df["second"].str.extract('(\d+)', expand=False).astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[158]:


#The first line removes any ".0" at the end of the string in the "Memory" column and converts the column to a string type.

#The next two lines remove the "GB" and "TB" units from the "Memory" column.

#The fourth line splits the "Memory" column at the "+" character, creating a new DataFrame "new" with two columns.

#The next four lines create four new columns in the original DataFrame based on the presence of certain strings in the "first" column of the "new" DataFrame.

#The next line removes any non-numeric characters from the "first" column of the "new" DataFrame.

#The next line fills any missing values in the "second" column of the "new" DataFrame with "0".

#The next four lines create four new columns in the original DataFrame based on the presence of certain strings in the "second" column of the "new" DataFrame.

#The next line removes any non-numeric characters from the "second" column of the "new" DataFrame.

#The next two lines convert the "first" and "second" columns to integer type.

#The next four lines calculate the total amount of HDD, SSD, Hybrid, and Flash Storage memory based on the values in the "first" and "second" columns and the presence of certain strings.

#The last line drops the columns that were created in steps 4 and 7, as they are no longer needed.


# In[159]:


df.drop(columns=['Memory','Flash_Storage','Hybrid'], inplace=True, axis=1)


# In[160]:


#df.corr()['Price']


# In[161]:


df['Gpu manufacturer']=df['Gpu'].apply(lambda x:x.split()[0])


# In[162]:


df.head()


# In[163]:


df=df[df['Gpu manufacturer']!= 'ARM']
df.drop(columns=['Gpu'],inplace=True)


# In[164]:


df['Gpu manufacturer'].value_counts()


# In[165]:


sns.barplot(x=df['Gpu manufacturer'], y=df['Price'],estimator=np.median )
#plt.xticks(rotation='45')
plt.show()


# In[166]:


sns.barplot(x=df['OpSys'], y=df['Price'] )
plt.xticks(rotation='vertical')

plt.show()


# In[167]:


df['OpSys'].unique()


# In[168]:


def OS_cat(value):
    if value=='Windows 10' or value=='Windows 7' or value=='Windows 10 S':
        return 'Windows'
    elif value=='macOS' or value=='Mac OS X' :
        return 'Mac'
    else:
        return 'Other/No OS/Linux'


# In[169]:


df['OS']=df['OpSys'].apply(OS_cat)


# In[170]:


df.head(1)


# In[171]:


df.columns


# In[172]:


df.drop(columns=['OpSys'],inplace=True)


# In[173]:


sns.distplot(np.log(df['Price']))


# In[174]:


df.head()


# In[175]:


df['Company'] = df['Company'].str.strip()


# In[176]:


X=df.drop(columns='Price', axis=1)
Y=np.log(df['Price'])


# In[177]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=34, train_size=0.8)


# In[178]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[179]:


from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[180]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe1 = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe1.fit(x_train,y_train)

y_pred = pipe1.predict(x_test)

print (f'Train Accuracy - : {pipe1.score(x_train,y_train)*100:.3f}')
print (f'Test Accuracy - : {pipe1.score(x_test,y_test)*100:.3f}')
print('MAE',mean_absolute_error(y_test,y_pred))


# In[181]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth= 8, min_samples_leaf= 3, min_samples_split= 4, random_state=42)

pipe2 = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe2.fit(x_train,y_train)

y_pred = pipe2.predict(x_test)

print (f'Train Accuracy - : {pipe2.score(x_train,y_train)*100:.3f}')
print (f'Test Accuracy - : {pipe2.score(x_test,y_test)*100:.3f}')
print('MAE',mean_absolute_error(y_test,y_pred))


# In[182]:


step1 = ColumnTransformer(transformers=[
 ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')
step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)
pipe3 = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe3.fit(x_train,y_train)

y_pred = pipe3.predict(x_test)

print (f'Train Accuracy - : {pipe3.score(x_train,y_train)*100:.3f}')
print (f'Test Accuracy - : {pipe3.score(x_test,y_test)*100:.3f}')
print('MAE',mean_absolute_error(y_test,y_pred))


# In[183]:


import pickle
pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe3,open('pipe.pkl','wb'))


# In[184]:


df


# In[185]:





# In[ ]:





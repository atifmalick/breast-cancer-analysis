#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


breast= pd.read_csv("F:\\Machine Learning\\Project\\Breast cancer\\breast_cancer.csv")
breast.head()


# In[3]:


breast['diagnosis'].value_counts()


# In[4]:


#Always ask some questions from the dataset first
breast.shape


# In[5]:


breast.isnull().sum()


# In[6]:


breast.duplicated().sum()


# In[7]:


breast.corr()


# In[8]:


breast.info()


# In[9]:


breast.drop('Unnamed: 32', axis=1, inplace=True)


# In[10]:


breast.describe()


# In[11]:


#Encoding 
#from sklearn.preprocessing import LabelEncoder
#le=LabelEncoder()


# In[12]:


breast['diagnosis']=breast['diagnosis'].map({'M':1,'B':0})


# In[13]:


X=breast.drop('diagnosis',axis=1)
y=breast['diagnosis']


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[17]:


sc.fit(X_train)


# In[18]:


X_train=sc.transform(X_train)
X_test=sc.transform(X_test)


# In[19]:


X_train


# In[20]:


from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()


# In[21]:


lg.fit(X_train,y_train)
y_pred=lg.predict(X_test)


# In[22]:


y_pred


# In[23]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[35]:


input_text=(6.63966343e+00,  1.97692750e+00,  3.13275910e-01,
         1.90233452e+00,  2.05786602e+00,  7.88036965e-02,
         2.64833778e-01,  8.60436091e-01,  1.76809679e+00,
        -9.35958109e-01, -8.53506343e-01,  9.27292155e-01,
         2.92548640e-01,  8.13824276e-01,  8.86282692e-01,
        -4.98473172e-01, -2.62852163e-01, -3.03227924e-02,
         4.39951802e-01, -1.25977636e+00, -5.80249898e-01,
         1.96523962e+00,  1.07854631e+00,  1.83265153e+00,
         2.01924985e+00,  2.28470287e-01,  3.70426070e-01,
         7.90300334e-01,  1.74421993e+00, -1.00983488e+00,
        -5.39471464e-01)
np_df=np.asarray(input_text)
prediciont=lg.predict(np_df.reshape(1,-1))

if prediciont[0]==1:
    print('cancrous')
else:
    print('non cancrous')


# In[34]:


X_train[11:20]


# In[36]:


import pickle
pickle.dump(lg,open('model.pkl','wb'))


# In[ ]:





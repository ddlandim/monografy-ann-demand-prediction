
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint

from sklearn import metrics
import os
get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir("../dados/input"))


# In[2]:


original_data = pd.read_csv('../dados/input/treino.csv')


# In[3]:


original_data.head()


# In[4]:


original_data.columns


# In[5]:


original_data.describe()


# In[6]:


original_data.info()


# In[8]:


plt.figure(figsize=(4,3))
sns.violinplot(x='Segunda',y='Total',data=original_data)
plt.figure(figsize=(4,3))
sns.violinplot(x='Terca',y='Total',data=original_data)
plt.figure(figsize=(4,3))
sns.violinplot(x='Quarta',y='Total',data=original_data)
plt.figure(figsize=(4,3))
sns.violinplot(x='Quinta',y='Total',data=original_data)
plt.figure(figsize=(4,3))
sns.violinplot(x='Sexta',y='Total',data=original_data)


# In[9]:


original_data.columns


# In[10]:


X_train = original_data.iloc[:,1:19]


# In[11]:


X_train.values


# In[12]:


Y_train = original_data.iloc[:,19].values


# In[13]:


Y_train


# In[14]:


test_data = pd.read_csv('../dados/input/teste.csv')


# In[15]:


test_data.head()


# In[16]:


Y_test = test_data.iloc[:,19].values


# In[17]:


Y_test


# In[18]:


X_test = test_data.iloc[:,1:19]


# In[19]:


X_test


# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)


# In[21]:


standardized_X_train


# In[22]:


standardized_X_test


# In[23]:


Y_train


# In[24]:


Y_test


# In[25]:


from keras import Sequential
from keras.layers import Dense

model = Sequential()

model.add(
            Dense(
                    standardized_X.shape[1],
                    activation='relu',
                    input_dim=standardized_X.shape[1]
            )
         )

model.add(
            Dense(1)
        )

model.compile(
                optimizer='rmsprop',
                loss='mse',
                metrics=['mse','mae','acc']
)


# In[26]:


from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=500,
 validation_split=0.3,,
 callbacks=[early_stopping_monitor])


# In[27]:


model.predict(standardized_X_test)


# In[28]:


score = model.evaluate(standardized_X_test,
y_test,
batch_size=36)


# In[29]:


score


# In[30]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=1000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[31]:


model.predict(standardized_X_test)


# In[32]:


score = model.evaluate(standardized_X_test,
y_test,
batch_size=36)


# In[33]:


score


# In[34]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=250,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[35]:


model.predict(standardized_X_test)


# In[36]:


score = model.evaluate(standardized_X_test,
y_test,
batch_size=36)


# In[37]:


score


# In[38]:


print(model.metrics_names)


# In[39]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=125,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[40]:


model.predict(standardized_X_test)


# In[41]:


score = model.evaluate(standardized_X_test,
y_test,
batch_size=36)


# In[42]:


score


# In[43]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=50,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[44]:


model.predict(standardized_X_test)


# In[45]:


score = model.evaluate(standardized_X_test,
y_test,
batch_size=36)


# In[46]:


score


# In[47]:


model = Sequential()
model.add(Dense(standardized_X.shape[1],activation='sigmoid',input_dim=standardized_X.shape[1]))
model.add(Dense(1))
model.compile(optimizer='rmsprop',
loss='mse',
metrics=['mae'])


# In[48]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=125,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[49]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=500,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[50]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=1000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[51]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=5000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[52]:


model = Sequential()
model.add(Dense(standardized_X.shape[1],activation='relu',input_dim=standardized_X.shape[1]))
model.add(Dense(1))
model.compile(optimizer='adam',
loss='mse',
metrics=['mae'])


# In[53]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=5000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[54]:


model = Sequential()
model.add(Dense(standardized_X.shape[1],activation='sigmoid',input_dim=standardized_X.shape[1]))
model.add(Dense(1))
model.compile(optimizer='adam',
loss='mse',
metrics=['mae'])


# In[55]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=5000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[56]:


model = Sequential()
model.add(Dense(standardized_X.shape[1],activation='relu',input_dim=standardized_X.shape[1]))
model.add(Dense(1))
model.compile(optimizer='adam',
loss='mse',
metrics=['mae'])


# In[57]:


model.fit(standardized_X,
 y_train,
 batch_size=36,
 epochs=5000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[58]:


model.fit(standardized_X,
 y_train,
 batch_size=148,
 epochs=5000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[59]:


model1 = Sequential()
model1.add(Dense(9,activation='relu',input_dim=standardized_X.shape[1]))
model1.add(Dense(1))
model1.compile(optimizer='adam',
loss='mse',
metrics=['mae'])


# In[60]:


model1.fit(standardized_X,
 y_train,
 batch_size=148,
 epochs=5000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[61]:


model2 = Sequential()
model2.add(Dense(standardized_X.shape[1],activation='relu',input_dim=standardized_X.shape[1]))
model2.add(Dense(9,activation='relu',input_dim=standardized_X.shape[1]))
model2.add(Dense(1))
model2.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[62]:


model2.fit(standardized_X,
 y_train,
 batch_size=148,
 epochs=5000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[63]:


model2.predict(standardized_X_test)


# In[64]:


score = model2.evaluate(standardized_X_test,
y_test,
batch_size=36)


# In[65]:


score


# In[66]:


model3 = Sequential()
model3.add(Dense(standardized_X.shape[1],activation='relu',input_dim=standardized_X.shape[1]))
model3.add(Dense(9,activation='relu',input_dim=standardized_X.shape[1]))
model3.add(Dense(3,activation='relu',input_dim=9))
model3.add(Dense(1))
model3.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[67]:


model3.fit(standardized_X,
 y_train,
 batch_size=148,
 epochs=5000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[68]:


model3.predict(standardized_X_test)


# In[69]:


score = model3.evaluate(standardized_X_test,
y_test,
batch_size=36)


# In[70]:


score


# In[71]:


model3 = Sequential()
model3.add(Dense(standardized_X.shape[1],activation='relu',input_dim=standardized_X.shape[1]))
model3.add(Dense(10,activation='relu',input_dim=standardized_X.shape[1]))
model3.add(Dense(5,activation='relu',input_dim=10))
model3.add(Dense(1))
model3.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[72]:


model3.fit(standardized_X,
 y_train,
 batch_size=148,
 epochs=5000,
 validation_data=(standardized_X_test,y_test),
 callbacks=[early_stopping_monitor])


# In[73]:


score = model3.evaluate(standardized_X_test,
y_test,
batch_size=36)


# In[74]:


score


# In[75]:


predictions1 = model.predict(standardized_X_test)
predictions2 = model2.predict(standardized_X_test)
predictions3 = model3.predict(standardized_X_test)


# In[76]:


plt.figure(figsize=(4,3))
plt.scatter(y_test[:200],predictions1[:200])
plt.xlabel('Y Test')
plt.ylabel('Model 1 Predicted Y')

plt.figure(figsize=(4,3))
plt.scatter(y_test[:200],predictions2[:200])
plt.xlabel('Y Test')
plt.ylabel('Model 2 Predicted Y')

plt.figure(figsize=(4,3))
plt.scatter(y_test[:200],predictions3[:200])
plt.xlabel('Y Test')
plt.ylabel('Model 3 Predicted Y')


# In[77]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions3))
print('MSE:', metrics.mean_squared_error(y_test, predictions3))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions3)))


# In[78]:


predictions3


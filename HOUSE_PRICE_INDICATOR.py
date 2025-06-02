#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


# In[13]:


df = pd.read_csv("house_prices.csv")


# In[4]:


df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)


# In[5]:


df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)


# In[6]:


X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[8]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)


# In[10]:



rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# In[11]:


print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# In[12]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





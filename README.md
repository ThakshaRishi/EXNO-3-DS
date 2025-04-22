## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("/content/Encoding_Data.csv")
df.head()
```

# Output
 ![image](https://github.com/user-attachments/assets/bcfc9c26-3fd7-4dcd-806a-c0cfe23aa9cf)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

# Output

![image](https://github.com/user-attachments/assets/11b2849d-a449-4cfb-bb72-8dbb5e715631)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

# Output

![image](https://github.com/user-attachments/assets/e2dfd00d-3475-486b-be24-b4b90e41d9c1)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

# Output

![image](https://github.com/user-attachments/assets/0314d318-de78-4d4b-aa3e-b78f880ae331)


```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

```
df2=pd.concat([df2,enc],axis=1)
df2
```

# Output

![image](https://github.com/user-attachments/assets/adace53c-bb46-4c50-b2ba-1294f38a59d2)

```
pd.get_dummies(df2,columns=["nom_0"])
```

# Output

![image](https://github.com/user-attachments/assets/e164ea17-029e-413a-9c0f-7119d0aaedaa)

```
pip install --upgrade category_encoders
```

# Output

![image](https://github.com/user-attachments/assets/34d804cf-4c36-4b68-8cda-8168ef16f8cb)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

# Output

![image](https://github.com/user-attachments/assets/7aecd142-9a2b-420d-b03e-b15b550af998)


```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

# Output

![image](https://github.com/user-attachments/assets/ced1f6b8-35f4-4c32-8866-0227e83904ef)


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```


# Output

![image](https://github.com/user-attachments/assets/83551532-f3b7-488b-b235-6935479f684b)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```


# Output

![image](https://github.com/user-attachments/assets/9af1eb31-2c5b-42ee-9849-ebb9ea66dc4d)


```
df.skew()
```

# Output

![image](https://github.com/user-attachments/assets/3c590268-4724-4087-a0b5-a2094d9e5bf2)

```

np.log(df["Highly Positive Skew"])
```

# Output

![image](https://github.com/user-attachments/assets/0596ac92-5ad4-41f2-af1a-3fefd049307d)

```

np.reciprocal(df["Moderate Positive Skew"])
```

# Output

![image](https://github.com/user-attachments/assets/e94c0bd4-ecb4-44c6-8523-c675937914b0)

```

np.sqrt(df["Highly Positive Skew"])
```

# Output

![image](https://github.com/user-attachments/assets/a94c2836-3ca2-470c-b57f-9ab1b071f042)

```

np.square(df["Highly Positive Skew"])
```

# Output

![image](https://github.com/user-attachments/assets/3e6266f4-d40a-47b2-86ff-123bf15d6a56)

```

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

# Output

![image](https://github.com/user-attachments/assets/95a1c735-cc21-43c3-9aab-8f49d49244ce)

```

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```

```

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

# Output

![image](https://github.com/user-attachments/assets/ee37ad98-9902-40dd-85c4-3d65bcbfa6ff)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

# Output

![image](https://github.com/user-attachments/assets/f34dead3-a9e7-4417-93b0-52d7c00f9ab4)



```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

# Output

![image](https://github.com/user-attachments/assets/5e156542-be37-4432-b101-1abe3ebeafaf)


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

# Output

![image](https://github.com/user-attachments/assets/41dc6d20-5d23-430e-b6e3-7ba1ccc68390)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

# Output

![image](https://github.com/user-attachments/assets/39caa1fd-5090-436b-bde0-be9d336661bb)


# RESULT:
      Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.


       

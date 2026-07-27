import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("insurance.csv")
print(df)
print(df.info())
# cleaning data
print(df.isnull().sum())
df=df.drop_duplicates()
print(df)
print(df.dtypes)
# EDA
# plt.figure(figsize =(8,5))

# sns.histplot(df["age"],bins=20,kde=True)

# plt.title("Age Distribution")

# plt.show()
# plt.figure(figsize=(8,5))

# sns.histplot(df["bmi"],bins=20,kde=True)

# plt.title("BMI Distribution")

# plt.show()

# plt.figure(figsize=(8,5))

# sns.histplot(df["charges"],bins=20,kde=True)

# plt.title("Insurance Charges Distribution")

# plt.show()
# sns.countplot(x="sex",data=df)

# plt.title("Gender Count")

# plt.show()

# sns.countplot(x="smoker",data=df)

# plt.title("Smoker Count")

# plt.show()

# sns.countplot(x="region",data=df)

# plt.title("Region Distribution")

# plt.show()



# plt.figure(figsize=(8,5))

# sns.boxplot(y=df["bmi"])

# plt.title("BMI Outliers")

# plt.show()



# plt.figure(figsize=(8,5))

# sns.boxplot(y=df["charges"])

# plt.title("Charges Outliers")

# plt.show()



# plt.figure(figsize=(8,5))

# sns.scatterplot(x="age",y="charges",data=df)

# plt.title("Age vs Charges")

# plt.show()



# plt.figure(figsize=(8,5))

# sns.scatterplot(x="bmi",y="charges",data=df,hue="smoker")

# plt.title("BMI vs Charges")

# plt.show()



# plt.figure(figsize=(8,5))

# sns.boxplot(x="smoker",y="charges",data=df)

# plt.title("Smoker vs Charges")



# plt.figure(figsize=(6,5))
# sns.boxplot(x="region", y="charges", data=df)
# plt.title("Region vs Insurance Charges")
# plt.show()

# plt.figure(figsize=(6,5))
# sns.boxplot(x="children", y="charges", data=df)
# plt.title("Children vs Insurance Charges")
# plt.show()

# plt.show()
# CORELATION OF ALL THE NUMARICAL SET
numeric_df=df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))

sns.heatmap(numeric_df.corr(),annot=True,cmap="coolwarm")

plt.title("Correlation Matrix")

plt.show()

categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]
df = pd.get_dummies(df, columns=["sex", "smoker", "region"])

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

df["bmi_category"] = df["bmi"].apply(bmi_category)
df = pd.get_dummies(df, columns=["bmi_category"], drop_first=True)

df["smoker_bmi"] = df["smoker_yes"] * df["bmi"]

X = df.drop("charges", axis=1)
y = df["charges"]
# print(X.head())
# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
prediction = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print(prediction.head(10))
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted")

plt.show()


original_df = pd.read_csv("insurance.csv")

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coef = coef.sort_values(by="Coefficient", ascending=False)

print(coef)

plt.figure(figsize=(10,5))
sns.barplot(data=coef, x="Coefficient", y="Feature")
plt.title("Feature Impact on Insurance Charges")
plt.show()

print(X.columns.tolist())
print(model.feature_names_in_)


import pickle
pickle.dump(model, open("model.pkl", "wb"))
loaded_model = pickle.load(open("model.pkl", "rb"))
sample = pd.DataFrame({
    "age":[30],
    "bmi":[28.5],
    "children":[2],

    "sex_female":[0],
    "sex_male":[1],

    "smoker_no":[1],
    "smoker_yes":[0],

    "region_northeast":[0],
    "region_northwest":[1],
    "region_southeast":[0],
    "region_southwest":[0],

    "bmi_category_Obese": [0],
    "bmi_category_Overweight": [1],
    "bmi_category_Underweight": [0],
    "smoker_bmi": [28.5]
})
print(X.columns.tolist())
prediction = loaded_model.predict(sample)

print("Predicted Insurance Cost:", prediction[0])

print(X.columns.tolist())
print(model.feature_names_in_)

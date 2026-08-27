import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Import and explore dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


# 2. Handle missing values

# Age → median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked → mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin has many missing values, so remove it
df.drop("Cabin", axis=1, inplace=True)


# 3. Convert categorical features into numerical values

df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True,
    dtype=int
)

print("\nAfter Encoding:")
print(df.head())


# 4. Visualize outliers using boxplots

numeric_cols = ["Age", "SibSp", "Parch", "Fare"]

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# Remove outliers using IQR

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]


# 5. Standardize numerical features

scale_cols = ["Age", "SibSp", "Parch", "Fare"]

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])


# Final result
print("\nFinal Dataset Shape:", df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nFinal Dataset:")
print(df.head())


# Save cleaned dataset
df.to_csv("Titanic-Cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
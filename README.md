# Task 1: Data Cleaning & Preprocessing

## 📌 Objective

The objective of this task is to learn how to clean and prepare raw data for Machine Learning by handling missing values, categorical data, outliers, and feature scaling.

## 🛠️ Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## 📊 Dataset

**Dataset:** Titanic Dataset

The Titanic dataset contains information about passengers, including their age, gender, passenger class, fare, and survival status.

## 🔍 Steps Performed

### 1. Import and Explore Dataset

The dataset was imported using Pandas and basic information was explored using:

* First few rows
* Dataset shape
* Data types
* Missing values

### 2. Handle Missing Values

Missing values were handled appropriately:

* `Age` → Missing values replaced using the **median**
* `Embarked` → Missing values replaced using the **mode**
* `Cabin` → Removed because it contained a large number of missing values

### 3. Categorical Data Encoding

Categorical features were converted into numerical values using **One-Hot Encoding**.

The following columns were encoded:

* `Sex`
* `Embarked`

### 4. Outlier Detection and Removal

Boxplots were used to visualize outliers in numerical features:

* Age
  <img width="551" height="390" alt="Screenshot 2026-08-27 211530" src="https://github.com/user-attachments/assets/64a709e9-2858-4282-a40c-e4c7dd4be9df" />
* SibSp
<img width="543" height="390" alt="Screenshot 2026-08-27 211540" src="https://github.com/user-attachments/assets/64c7dabc-9a09-415d-bac9-15975b300c71" />
* Parch
  <img width="530" height="383" alt="Screenshot 2026-08-27 211550" src="https://github.com/user-attachments/assets/2148f08a-1216-4074-981f-f48e1295b59f" />
* Fare
  <img width="548" height="392" alt="Screenshot 2026-08-27 211611" src="https://github.com/user-attachments/assets/8ae56377-b745-4cac-b312-3212b1e37899" />

The **IQR (Interquartile Range)** method was then used to identify and remove outliers.

### 5. Feature Standardization

Numerical features were standardized using **StandardScaler** so that the features have a comparable scale.

The following features were standardized:

* Age
* SibSp
* Parch
* Fare

### 6. Save Cleaned Dataset

After preprocessing, the cleaned dataset was saved as:

`Titanic-Cleaned.csv`

## 📁 Project Structure

```text
TASK-1_Data-Cleaning/
│
├── data_cleaning.py
├── Titanic-Dataset.csv
├── Titanic-Cleaned.csv
└── README.md
```

## ✅ Conclusion

The Titanic dataset was successfully cleaned and preprocessed for Machine Learning. Missing values were handled, categorical features were encoded, outliers were detected and removed, and numerical features were standardized. The resulting cleaned dataset can now be used for further Machine Learning analysis and model building.

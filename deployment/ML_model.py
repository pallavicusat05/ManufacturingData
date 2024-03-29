# -*- coding: utf-8 -*-
"""Manufacturing Classification Dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F0hsRkdE9eF-AdMnWbVnQxZD73jzgycC

**This is the data from a semiconductor manufacturing process. We will analyze whether all the features are required to build the model or not and build a classifier to predict the Pass/Fail yield of a particular process entity.**

# Machine Learning Approach
---

# Exploratory Data Analysis
"""

# Commented out IPython magic to ensure Python compatibility.
# Importing necessary libraries
import joblib
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
sns.set()
import warnings

warnings.filterwarnings('ignore')
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay, \
    RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, roc_curve

# Reading the data from csv file and checking first five rows
df = pd.read_csv('data/manufacturing_dataset.csv')
df.head()

df.info()

print(df.shape)
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns\n')

# Renaming the features columns to features_<number> instead of <number>
df.columns = 'features_' + df.columns

df.rename(columns={'features_Time': 'Time'}, inplace=True)
df.rename(columns={'features_Pass/Fail': 'Pass/Fail'}, inplace=True)

"""# Data Preprocessing

Dropping 'Time' column, as it does not provide much value
"""

df = df.drop(['Time'], axis=1)

"""Since there is lot of extra columns (features) which has missing values and many of them might not make sense, so need to drop them."""

# There are many features with more than 40% missing values. Those will not provide much insight in prediction, so needs to be dropped
def remove_null_columns(data, thres):
    columns = data.columns
    cols_remove = []
    for i in columns:
        if (data[i].isna().sum() / data.shape[0] >= thres):
            cols_remove.append(i)
    print(f'Number of features removed with more than {thres}% of null values : \t {len(cols_remove)}')
    data = data.drop(labels=cols_remove, axis=1)
    return (data)


df = remove_null_columns(df, 0.4)

# There are many columns with single unique value. These columns need to be dropped as well, as they provide value in classification
uni_list = []
for column in df.columns:
    if (df[column].nunique() == 1):
        uni_list.append(column)
print(f'Number of features with single unique value removed : \t {len(uni_list)}')
df.drop(columns=uni_list, axis=1, inplace=True)

# We can select highly correlated features with the following function
# It will remove the first feature that is correlated with any other feature
def correlation(dataframe, threshold):
    # Set of all the names of correlated columns
    col_corr = set()
    corr_matrix = dataframe.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # Getting the absolute correlation coefficient value
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Getting the name of the first feature that is correlated with any other feature
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# Removing features having more than 70% correlation from the original DataFrame
# Both positive and negative correlations are considered here
corr_features = correlation(df, 0.7)
df = df.drop(corr_features, axis=1)
print(f'After removing {len(corr_features)} correlated features, there are {df.shape[1]} features left.')

# There are several highly multicollinear features. We can remove these features as well.
# It will get the list of all the highly multicollinear features having VIF greater than vif_threshold
def high_vif_features(dataframe, vif_threshold):
    dataframe = dataframe.dropna()
    # Create a DataFrame to store VIF values and their corresponding features
    vif_data = pd.DataFrame()
    vif_data["feature"] = dataframe.columns
    vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    # Find features with VIF greater than the threshold
    high_vif_columns = vif_data[vif_data["VIF"] > vif_threshold]["feature"].tolist()
    return high_vif_columns


# Removing high VIF features (having VIF greater than 10) from the original DataFrame
high_vif_cols = high_vif_features(df, 10)
df = df.drop(high_vif_cols, axis=1)
print(f'After removing {len(high_vif_cols)} highly multicollinear features, there are {df.shape[1]} features left.')

# We would like our features to have high correlation with the target.
# If a feature has low correlation with target, it means that it is not a helpful feature for predicting the target, and hence, should be removed.
# It will remove the features having low correlation (less than threshold) with the target
def corr_with_target(dataframe, target, threshold):
    cor = dataframe.corr()
    # Correlation with output variable
    cor_target = abs(cor[target])
    # Selecting non correlated features
    relevant_features = cor_target[cor_target < threshold]
    return relevant_features.index.tolist()[:-1]


# Removing features having low correlation (less than 5%) with the target ('Pass/Fail')
corr_cols = corr_with_target(df, 'Pass/Fail', 0.05)
df = df.drop(corr_cols, axis=1)
print(
    f'After removing {len(corr_cols)} features having low correlation with target, there are {df.shape[1]} features left.')

"""
**Missing value Imputation**
"""
# checking for missing values
df.isnull().sum()

# Since the columns are numeric, replacing missing values with the median for each column
df = df.apply(lambda col: col.fillna(col.median()))

df.isnull().sum()

"""**Feature Encoding**"""

# There is no encoding required, as all the feature columns are float.
# Changing the value of target column, so that each failure is encoded as 0 and 1 corresponds to a pass, for easy understanding
df['Pass/Fail'] = df['Pass/Fail'].replace(to_replace=1, value=0)
df['Pass/Fail'] = df['Pass/Fail'].replace(to_replace=-1, value=1)

# Splitting the data into independent and dependent variables
x = df.drop(['Pass/Fail'], axis=1)
y = df['Pass/Fail']

# Splitting the data into train (used for training the model) and test (used for validating the model)
# We have splitted as 80% training data and 20% test data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=101, test_size=0.2, stratify=y)
# Stratify is used to divide the dataset such that both train and test data have representation from both the classes (0 and 1) in the same proportion as it is available in the original dataset

x_train.head()

"""**Feature Scaling**"""

# Feature Scaling required, as there is huge difference in the magnitude of values from one column to other
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train = pd.DataFrame(x_train_scaled, columns=x.columns[:])
x_train.head()

x_test = pd.DataFrame(x_test_scaled, columns=x.columns[:])
x_test.head()

"""**Imbalance Treatment**"""

# Imbalance treatement required, since target class is highly imbalanced
# Using SMOTE technique to balance the target classes
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x_train, y_train)
print("Before Smote data counts : \n", y_train.value_counts())
print("After somte data counts : \n", y_smote.value_counts())

"""# Model Building"""

model = LogisticRegression()
model.fit(x_train , y_train)

# Saving the model to pickle file for deployment
joblib.dump(model, 'Stacking_model.pkl')

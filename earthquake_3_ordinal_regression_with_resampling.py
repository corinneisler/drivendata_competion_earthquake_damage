# Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, f_regression
from mord import LogisticAT
from imblearn.combine import SMOTETomek

## DATA PREPARATION

# Read the csv files
X_train = pd.read_csv('https://s3.amazonaws.com/drivendata/data/57/public/train_values.csv', index_col= 'building_id')
y_train = pd.read_csv('https://s3.amazonaws.com/drivendata/data/57/public/train_labels.csv', index_col= 'building_id')
X_test = pd.read_csv('https://s3.amazonaws.com/drivendata/data/57/public/test_values.csv', index_col= 'building_id')

# Transform geo_level_1_id from int to string, drop geo_level_2_id and geo_level_3_id (it does not make sense to treat geo level ids as integers)
X_train['geo_level_1_id'] = X_train[['geo_level_1_id']].astype(str)
X_test['geo_level_1_id'] = X_test[['geo_level_1_id']].astype(str)
X_train = X_train.drop(columns = ['geo_level_2_id','geo_level_3_id'])
X_test = X_test.drop(columns = ['geo_level_2_id','geo_level_3_id'])

# Drop unnecessary columns (based on preliminary analysis in file 1: preprocessing)

# Drop all columns containing "has_secondary_use" (very uneven frequency distribution and low relevance)
X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='has_secondary_use')))]

# Drop all columns of dummy variables where one value has 95% or more frequency
X_train = X_train[X_train.columns.drop(['has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other'])]

# Drop the variable count_floors_pre_eq
X_train = X_train[X_train.columns.drop(['count_floors_pre_eq'])]

# All land surface conditions and positions have very similar distributions of damage grade (visible in the created crosstab visualizations) -> drop these variables
X_train = X_train.drop(columns=['land_surface_condition', 'position'])

# Drop the same columns for X_test dataset
X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='has_secondary_use')))]
X_test = X_test[X_test.columns.drop(['has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other'])]
X_test = X_test[X_test.columns.drop(['count_floors_pre_eq'])]
X_test = X_test.drop(columns=['land_surface_condition', 'position'])

# Create dummies
dummies_train = pd.get_dummies(X_train[['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status']], prefix = ['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',  'plan_configuration','legal_ownership_status'])
dummies_test = pd.get_dummies(X_test[['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status']], prefix = ['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',  'plan_configuration','legal_ownership_status'])

# Merge the original dataframes with the newly created dummy dataframes, drop the original categorical variables
X_train = pd.concat([X_train, dummies_train], axis=1)
X_train = X_train.drop(columns=['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status'])

X_test = pd.concat([X_test, dummies_test], axis=1)
X_test = X_test.drop(columns=['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status'])

# Transform the independent variables, so that all lie between 0 and 1 (normalization)
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_train = pd.DataFrame(scaled_X_train, columns = X_train.columns)

## FEATURE SELECTION

# SelectKBest (try between 1 and 20 features, check which number of features leads to the best f1 score)
f1_list = []
for k in range(1, 21):
    bk = SelectKBest(f_regression, k=k)
    bk.fit(scaled_X_train, y_train.values.ravel())
    X_trans = bk.transform(scaled_X_train)
    X_train_train, X_test_train, y_train_train, y_test_train = train_test_split(X_trans,
                                                        y_train,
                                                        test_size=0.3,
                                                        random_state=42)
    regressor = LogisticAT(alpha=1.0, verbose=0)
    regressor.fit(X_train_train, y_train_train.values.ravel())
    y_pred = regressor.predict(X_test_train)
    f1 = f1_score(y_test_train, y_pred, average='micro')
    f1_list.append(f1)
print(f1_list)
# Best f1 score (0.6584): 20 features

# Feature Selection: SelectKBest
bestfeatures=SelectKBest(f_regression, k=30)
fit=bestfeatures.fit(scaled_X_train, y_train.values.ravel())
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(scaled_X_train.columns)
featurescore=pd.concat([dfcolumns,dfscores],axis=1)
featurescore.columns=['Specs','Score']
print(featurescore.nlargest(30,'Score'))

# Dataframe with only the 20 best features
best20 = featurescore.nlargest(20, 'Score')
columns = best20['Specs'].values.tolist() # Export all selected feature as a list
X_train_best20 = scaled_X_train[columns] # New dataframe with only the 20 selected features in the columns
print(X_train_best20.info())

# Select the same 20 features for X_test as well
X_test_best20 = X_test[columns] # New dataframe with only the 20 selected features in the columns

## RESAMPLING

# Oversampling followed by undersampling (SMOTETomek)
columns = X_train_best20.columns
smt = SMOTETomek(random_state=0)
X_smt, y_smt = smt.fit_resample(X_train_best20, y_train.values.ravel())

## ORDINAL REGRESSION

ordreg = LogisticAT(alpha=1.0, verbose=0)

# Fit the model
ordreg.fit(X_smt, y_smt)

# Predict y_train based on X_train_best20 (predictions), compare to actual y_train values
predictions = ordreg.predict(X_smt)
f1_score_micro = f1_score(y_smt, predictions, average='micro')
print('Micro-averaged F1 score:', f1_score_micro)
# Micro-averaged F1 score: 0.6108

# Confusion matrix
cm = confusion_matrix(y_smt, predictions)

# Visualize confusion matrix
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', cmap='Greens', ax = ax)

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['1', '2', '3'])
ax.yaxis.set_ticklabels(['1', '2', '3'])

# Fix for mpl bug that cuts off top/bottom boxes of heatmap
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b,t) # update the ylim(bottom, top) values
plt.tight_layout()
plt.show()

'''
The confusion matrix looks quite similar to the one for the multinomial logistic regression model with resampling. However, the overprediction of
damage grade 2 is stronger here than in the logistic regression model. At the same time, it is still not as strong as for the models with no
resampling.
'''

# Predict y_test
y_pred = ordreg.predict(X_test_best20)

# Create a dataframe with 2 columns (building_id and the predicted damage grade values)
X_test_best20 = X_test_best20.reset_index()
df = pd.concat([X_test_best20['building_id'], pd.DataFrame(y_pred)],axis=1) # Concatenate building id column from X_test dataframe with the newly created array of y_test predictions
df.rename(columns = {0:'damage_grade'}, inplace = True)
print(df.head())

# Export y_pred dataframe to csv file
df.to_csv("y_pred_3_ordinal_regression_selectKBest20_with_resampling.csv", index=False)

'''
Submission do Driven Data competition:
Micro-averaged F1 Score: 0.6115

This model performs rather poorly in predicting the test labels. Better than the logistic regression model with resampling, but considerably
worse than the logistic regression model without resampling.

without resampling, the ordinal regression model only predicts damage grades 2 and 3 -> not submitted to competion at all

'''

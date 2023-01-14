# Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_regression

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

# Create dummies
dummies_train = pd.get_dummies(X_train[['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status']], prefix = ['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',  'plan_configuration','legal_ownership_status'])
dummies_test = pd.get_dummies(X_test[['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status']], prefix = ['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',  'plan_configuration','legal_ownership_status'])

# Merge the original dataframes with the newly created dummy dataframes, drop the original categorical variables
X_train = pd.concat([X_train, dummies_train], axis=1)
X_train = X_train.drop(columns=['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status', 'land_surface_condition', 'position'])

X_test = pd.concat([X_test, dummies_test], axis=1)
X_test = X_test.drop(columns=['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status', 'land_surface_condition', 'position'])

## FEATURE SELECTION

# Feature Selection: SelectKBest (with f_regression)
bestfeatures=SelectKBest(f_regression, k=30)
fit=bestfeatures.fit(X_train, y_train.values.ravel())
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X_train.columns)
featurescore=pd.concat([dfcolumns,dfscores],axis=1)
featurescore.columns=['Specs','Score']
print(featurescore.nlargest(30,'Score'))

# Dataframe with only the 20 best features (tried out also models with best 10, 12, 15 and 18 features)
best20 = featurescore.nlargest(20, 'Score')
columns = best20['Specs'].values.tolist() # Export all selected feature as a list
X_train_best20 = X_train[columns] # New dataframe with only the 20 selected features in the columns
print(X_train_best20.info())

# Select the same 20 features for X_test as well
X_test_best20 = X_test[columns] # New dataframe with only the 20 selected features in the columns

'''
## RESAMPLING

# Oversampling followed by undersampling (SMOTETomek)
columns = X_train_best20.columns
smt = SMOTETomek(random_state=0)
X_smt, y_smt = smt.fit_resample(X_train_best20, y_train.values.ravel())

Results are better without resampling than with it -> not using resampling for this model
'''

## RANDOM FOREST MODEL

# Pipeline with scaler and model
pipe = make_pipeline(StandardScaler(),
                     RandomForestClassifier(random_state=2018))

# Try out several parameters each for n_estimators and min_samples_leaf
param_grid = {'randomforestclassifier__n_estimators': [100,120, 150],
              'randomforestclassifier__min_samples_leaf': [5, 10, 15]}

# GridSearchCV with cross validation
gs = GridSearchCV(pipe, param_grid, cv=5)

# Fit the model
gs.fit(X_train_best20, y_train.values.ravel())

# Print the best parameters found by GridSearchCV (from those that are in the param_grid)
gs.best_params_
print("Best parameters:", gs.best_params_)
# Best parameters:
# n_estimators: 5, min_samples_leaf: 150

# Predict y_train based on X_train_best20 (predictions), compare to actual y_train values
predictions = gs.predict(X_train_best20)
f1_score_micro = f1_score(y_train, predictions, average='micro')
print('Micro-averaged F1 score:', f1_score_micro)
# Micro-averaged F1 score: 0.6559

# Confusion matrix
cm = confusion_matrix(y_train, predictions)

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
The model overpredicts damage grade 2, especially for cases where the true label is damage grade 3.
'''

# Predict y_test based on X_test_best20
y_pred = gs.predict(X_test_best20)

# Create a dataframe with 2 columns (building_id and the predicted damage grade values)
X_test_best20 = X_test_best20.reset_index()
df = pd.concat([X_test_best20['building_id'], pd.DataFrame(y_pred)],axis=1) # Concatenate building id column from X_test dataframe with the newly created array of y_test predictions
df.rename(columns = {0:'damage_grade'}, inplace = True)
print(df.head())

# Export y_pred dataframe to csv file
df.to_csv("y_pred_1_random_forest_selectKBest20_no_resampling.csv", index=False)

'''
Submission do Driven Data competition:
Micro-averaged F1 Score: 0.6492
'''

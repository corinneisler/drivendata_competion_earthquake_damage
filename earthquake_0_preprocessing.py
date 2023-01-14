# Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Read the csv files
X_train = pd.read_csv('https://s3.amazonaws.com/drivendata/data/57/public/train_values.csv', index_col= 'building_id')
y_train = pd.read_csv('https://s3.amazonaws.com/drivendata/data/57/public/train_labels.csv', index_col= 'building_id')
X_test = pd.read_csv('https://s3.amazonaws.com/drivendata/data/57/public/test_values.csv', index_col= 'building_id')

# Relative value counts of the target variable (damage grade)
print(y_train.damage_grade.value_counts(normalize=True))

# Plot the frequency distribution of damage grade
ax = y_train['damage_grade'].value_counts().sort_index().plot(kind='bar', color='#368ce7')
plt.title('Frequency distribution of damage grade', fontsize=16, pad=20)
plt.xlabel('Damage grade', fontsize=14, labelpad=15)
plt.ylabel('Frequency count', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
ax.yaxis.grid(color='#99A3A4', alpha=0.8, linewidth=0.5)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height()/2, \
            str(round((bar.get_height()/len(y_train.index))*100, 2))+'%', ha='center', va='center', fontsize=13,
                color='white')
plt.show()

# Print variable types
print(X_train.info())

# Descriptive statistics of all independent variables
with pd.option_context('display.max_columns', 40):
    print(X_test.describe(include='all'))

'''
Types of independent variables in the dataset:

8 categorical variables, 9 continuous variables, 22 binary variables
'''
# Drop unnecessary binary variables

# Frequency counts for all binary variables (percentages)
for col in X_train.columns:
    if (len(X_train[col].unique()) == 2):
      print(X_train[col].value_counts(normalize=True))

# Drop all columns containing "has_secondary_use" (very uneven frequency distribution and low relevance)
X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='has_secondary_use')))]
X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='has_secondary_use')))]


# Drop all columns of dummy variables where one value has 95% or more frequency
X_train = X_train[X_train.columns.drop(['has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other'])]
X_test = X_test[X_test.columns.drop(['has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other'])]

# Calculate Spearman correlation coefficients between all remaining variables
corr1 = round(X_train.corr(method='spearman'),2)
print(corr1)

# Heatmap of Spearman correlations
sns.set(style='ticks', color_codes=True)
plt.figure(figsize=(18, 15))

# Use mask to only show half of the heatmap (lower triangle)
mask = np.zeros_like(X_train.corr(method='spearman'))
mask[np.triu_indices_from(mask)] = True

sns.heatmap(round(X_train.corr(method='spearman'),2),
            mask = mask,
            cmap = 'Blues',
            linewidths=0.1,
            square=True,
            linecolor='white',
            annot=True)

# Fix for mpl bug that cuts off top/bottom boxes of heatmap
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.tight_layout()
plt.show()

# height_percentage and count_floors_pre_eq are (understandbly) strongly correlated -> One of these variables can be dropped

# Drop the variable count_floors_pre_eq
X_train = X_train[X_train.columns.drop(['count_floors_pre_eq'])]
X_test = X_test[X_test.columns.drop(['count_floors_pre_eq'])]

# Crosstabs of categorical variables with damage grade

# Merge the training datasets into one dataset (df)
df = pd.merge(X_train, y_train, left_index=True, right_index=True)

ct_foundation = pd.crosstab(df.foundation_type, df.damage_grade, normalize='index')
print(ct_foundation)

ax = ct_foundation.plot.bar(stacked=True, color=['#bedaf7', '#368ce7', '#20397d'])
plt.title('Damage grade by foundation type', fontsize=16, pad=20)
plt.xlabel('Foundation type', fontsize=14, labelpad=15)
plt.ylabel('Frequency', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),title='Damage grade', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

ct_groundfloor = pd.crosstab(df.ground_floor_type, df.damage_grade, normalize='index')
print(ct_groundfloor)

ax = ct_groundfloor.plot.bar(stacked=True, color=['#bedaf7', '#368ce7', '#20397d'])
plt.title('Damage grade by ground floor type', fontsize=16, pad=20)
plt.xlabel('Ground floor type', fontsize=14, labelpad=15)
plt.ylabel('Frequency', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),title='Damage grade', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

ct_roof = pd.crosstab(df.roof_type, df.damage_grade, normalize='index')
print(ct_roof)

ax = ct_roof.plot.bar(stacked=True, color=['#bedaf7', '#368ce7', '#20397d'])
plt.title('Damage grade by roof type', fontsize=16, pad=20)
plt.xlabel('Roof type', fontsize=14, labelpad=15)
plt.ylabel('Frequency', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),title='Damage grade', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

ct_other_floor = pd.crosstab(df.other_floor_type, df.damage_grade, normalize='index')
print(ct_other_floor)

ax = ct_other_floor.plot.bar(stacked=True, color=['#bedaf7', '#368ce7', '#20397d'])
plt.title('Damage grade by other floor type', fontsize=16, pad=20)
plt.xlabel('Other floor type', fontsize=14, labelpad=15)
plt.ylabel('Frequency', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),title='Damage grade', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

ct_land_surface = pd.crosstab(df.land_surface_condition, df.damage_grade, normalize='index')
print(ct_land_surface)

ax = ct_land_surface.plot.bar(stacked=True, color=['#bedaf7', '#368ce7', '#20397d'])
plt.title('Damage grade by land surface condition', fontsize=16, pad=20)
plt.xlabel('Land surface condition', fontsize=14, labelpad=15)
plt.ylabel('Frequency', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),title='Damage grade', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

ct_position = pd.crosstab(df.position, df.damage_grade, normalize='index')
print(ct_position)

ax = ct_position.plot.bar(stacked=True, color=['#bedaf7', '#368ce7', '#20397d'])
plt.title('Damage grade by position', fontsize=16, pad=20)
plt.xlabel('Position', fontsize=14, labelpad=15)
plt.ylabel('Frequency', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),title='Damage grade', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

ct_plan_configuration = pd.crosstab(df.plan_configuration, df.damage_grade, normalize='index')
print(ct_plan_configuration)

ax = ct_plan_configuration.plot.bar(stacked=True, color=['#bedaf7', '#368ce7', '#20397d'])
plt.title('Damage grade by plan configuration', fontsize=16, pad=20)
plt.xlabel('Plan configuration', fontsize=14, labelpad=15)
plt.ylabel('Frequency', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),title='Damage grade', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

ct_legal_ownership_status = pd.crosstab(df.legal_ownership_status, df.damage_grade, normalize='index')
print(ct_legal_ownership_status)

ax = ct_legal_ownership_status.plot.bar(stacked=True, color=['#bedaf7', '#368ce7', '#20397d'])
plt.title('Damage grade by legal ownership status', fontsize=16, pad=20)
plt.xlabel('Legal ownership status', fontsize=14, labelpad=15)
plt.ylabel('Frequency', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),title='Damage grade', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

ct_geo_level_1_id = pd.crosstab(df.geo_level_1_id, df.damage_grade, normalize='index')
print(ct_geo_level_1_id)

ax = ct_geo_level_1_id.plot.bar(stacked=True, color=['#bedaf7', '#368ce7', '#20397d'])
# plt.figure(figsize=(6, 12))
plt.title('Damage grade by Geo level 1 id', fontsize=16, pad=20)
plt.xlabel('Geo level id', fontsize=14, labelpad=15)
plt.ylabel('Frequency', fontsize=14, labelpad=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),title='Damage grade', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# All land surface conditions and positions have very similar distributions of damage grade (visible in the created crosstab visualizations) -> drop these variables
X_train = X_train.drop(columns=['land_surface_condition', 'position'])
X_test = X_test.drop(columns=['land_surface_condition', 'position'])

# Transform geo_level_1_id from int to string, drop geo_level_2_id and geo_level_3_id (it does not make sense to treat geo level ids as integers)
X_train['geo_level_1_id'] = X_train[['geo_level_1_id']].astype(str)
X_test['geo_level_1_id'] = X_test[['geo_level_1_id']].astype(str)
X_train = X_train.drop(columns = ['geo_level_2_id','geo_level_3_id'])
X_test = X_test.drop(columns = ['geo_level_2_id','geo_level_3_id'])

# Create dummy variables

# Transform the 6 remaining categorical variables into dummy variables
dummies_train = pd.get_dummies(X_train[['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status']], prefix = ['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',  'plan_configuration','legal_ownership_status'])
dummies_test = pd.get_dummies(X_test[['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status']], prefix = ['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',  'plan_configuration','legal_ownership_status'])

# Merge X_train and X_test with the newly created dummy dataframes
X_train = pd.concat([X_train, dummies_train], axis=1)
X_test = pd.concat([X_test, dummies_test], axis=1)

# Drop the original categorical variables
X_train = X_train.drop(columns=['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status'])
X_test = X_test.drop(columns=['geo_level_1_id','foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'plan_configuration','legal_ownership_status'])

# Also drop those dummy variables where the distribution of damage grade is very similar to the one in the overall data (visible in crosstabs above)
X_train = X_train.drop(columns=['foundation_type_u', 'foundation_type_w', 'ground_floor_type_f', 'ground_floor_type_x', 'roof_type_n', 'roof_type_q', 'other_floor_type_q', 'other_floor_type_x', 'plan_configuration_a', 'plan_configuration_c', 'plan_configuration_o', 'plan_configuration_s', 'plan_configuration_u', 'legal_ownership_status_r', 'legal_ownership_status_v'])
X_test = X_test.drop(columns=['foundation_type_u', 'foundation_type_w', 'ground_floor_type_f', 'ground_floor_type_x', 'roof_type_n', 'roof_type_q', 'other_floor_type_q', 'other_floor_type_x', 'plan_configuration_a', 'plan_configuration_c', 'plan_configuration_o', 'plan_configuration_s', 'plan_configuration_u', 'legal_ownership_status_r', 'legal_ownership_status_v'])

# Check that all independent variables are numerical now
print(X_train.info())
print(X_test.info())

# Scale the independent variables

# Transform the independent variables, so that all lie between 0 and 1 (normalization)
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_train = pd.DataFrame(scaled_X_train, columns = X_train.columns)

with pd.option_context('display.max_columns', 30):
    print(scaled_X_train.describe(include='all'))

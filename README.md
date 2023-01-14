RICHTER'S PREDICTOR: MODELING EARTHQUAKE DAMAGE ON DRIVENDATA

1. Problem description:

The goal of this data science competition by DrivenData is to predict the severity of damages to buildings as a consequence of the 2015 Ghorka earthquake in Nepal: https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/

There are 3 grades of damage in the dataset:

      1: low damage
  
      2: medium amount of damage
  
      3: almost complete destruction

The dataset contains information about each building’s structure, location and legal ownership (38 features in total). 

To measure the performance of the models, the micro-averaged F1 score is used which balances precision and recall of a classifier. 

2. Preprocessing of the data

The distribution of the dependent variable damage is somewhat uneven, with the most frequent category (damage grade 2) consisting of 57% of the observations (Figure 0a). 

After a preliminary analysis of the independent variables by means of a correlation heatmap and crosstabs (Figures 0b-0k), some variables are dropped from the analysis due to either being highly correlated to another variable or having a very similar distribution of damage grades as the overall data. 

The geographical information in the original dataset consists of three levels of geographical areas. These geographical ids (geo_level_1_id, geo_level_2_id, geo_level_3_id) are of type integer. For the purpose of this analysis it does however not make sense to treat these variables as integers. For this reason, I transform the highest geographical level (geo_level_1_id) into a categorical variable, which is included with the other categorical variables when creating dummy variables. I decided to completely drop the other two geographic variables from my models and focus more on the variables that describe buildings’ structure and ownership status. An alternative method would be using target encoding for the geographical variables, as for example used here: https://becominghuman.ai/how-to-shine-in-a-data-science-challenge-an-example-from-drivendata-47a526fa38ea 

3. Models

3.1 Random forest

The benchmark model for the DrivenData competition is a random forest model (F1 Score: 0.5815), so I decided to try this as my first model and improve it. After dropping the variables deemed unnecessary during preprocessing, I apply SelectKBest feature selection to reduce the number of features. After trying out different numbers of features, I decided on 20 features for this model. 

With some hyperparameter tuning, the micro-averaged F1 Score for the test data is 0.6492. While this score is a clear improvement over the one of the benchmark model, the confusion matrix for the training data (Figure 1) shows that the model strongly overpredicts damage grade 2, in particular for cases where the true label is damage grade 3. Despite the frequency distribution of damage grade being somewhat imbalanced, using resampling (SMOTETomek) considerably worsened rather than improved the performance of this model. 

In addition, even though the performance in terms of the F1 Score could most likely still be improved for this model, random forest models are hard to interpret which is a clear disadvantage for this type of classification problem. In this case, it is not only  important to understand which features are having the strongest influence on the damage but also whether this influence is positive or negative. 

3.2 Multinomial logistic regression

The second model I tried is a multinomial logistic regression model, even though the dependent variable is ordinal rather than categorical and thus some information is lost when applying this model.

Also here, the number of features selected by means of SelectKBest is 20 features. By means of a loop and train-test-split on the training data which checks the F1 Scores for models with 1-20 features, the version with 20 features turned out to be the best option. I decided against including more than 20 features in the model due to interpretability concerns, even if such a model might perform somewhat better than this current one. 

  a ) Without resampling

Without resampling the data, the logistic regression model achieved an F1 score of 0.6584 for the test data, a slight improvement over the random forest model. However, this model also strongly overpredicts damage grade 2 for the training data (Figure 2a). 

  b) With SMOTETomek resampling

As the frequency distribution of the target variable is somewhat imbalanced, I decided to train an alternative logistic regression model in which I use SMOTETomek resampling after the feature selection. Unfortunately, the performance of this model is rather poor with an F1 Score of 0.5806 for the test data, which is lower than the score of the benchmark model of the competition. 
Nonetheless, it should be noted that the prediction of damage grades 1 and 3 (Figure 2b) is considerably better for this model than it is for the one without resampling. One could also argue that predicting the highest and lowest grade correctly is more important than predicting the medium damage grade correctly. 

3.3 Ordinal regression

As already mentioned, using a multinomial logistic regression model for this problem leads to a loss of information, as the target variable is ordinal. Also for this model, I use the 20 best features, identified by means of SelectKBest, analogously to the logistic regression model above. 

Without resampling, this model exclusively predicts damage grades 2 and 3 and was therefore not submitted to the competition at all. With SMOTETomek resampling, it reaches an F1 Score of 0.6115 for the test data. This is better than the logistic regression model with resampling, but unfortunately still clearly lower than the logistic regression without resampling. The confusion matrix looks similar to the logistic regression model with resampling, but the ordinal regression model overpredicts damage grade 2 more strongly (Figure 3). 

3.4 Decision tree classifier

I also decided to try out a decision tree classifier, as the interpretability should be better than for the random forest model. Like the other models, this one also uses the 20 best features chosen by means of SelectKBest. 

Like the ordinal regression model without resampling, this model also only predicts damage grades 2 and 3 for the test data when no resampling is used. With resampling, the F1 Score is 0.5797, which is the worst performance of all the models presented here. In addition, a decision tree with 20 features is also rather hard to interpret. 

4. Conclusion and next steps

After looking at the performance of each of the models, I also want to have a look at the contribution of the individual variables. The two tables in Figure 5 show the coefficients of the logistic regression model for damage grade 2 and 3 respectively (both using damage grade 1 as the reference category). It can be seen for example that foundation type r is associated with relatively high grades of damage, while types w and u are more frequently found in buildings with medium damage. Roof type x and foundation type i are uncommon among buildings with damage grade 3, for example.
The competition does not contain any information regarding what the different categories of roof types, ground floor types etc. are in reality. 

While the multinomial logistic regression model without resampling had the best F1 Score of all the models here (0.6584) and is easier to interpret than the decision tree and random forest models, it does have significant weaknesses, particularly the strong overprediction of damage grade 2. In a next step it would make most sense to try out different feature selection strategies, as choosing the best features is the most important part for a model that should not only perform well but also be interpretable. 

Also different resampling techniques are of course with trying, I would however refrain from investing too much time into hyperparameter tuning even though this could of course also improve the F1 Score. As already mentioned above, target encoding of the geographical variables might also be a more fruitful approach than dropping the lower two levels and treating the highest level as a categorical variable. 


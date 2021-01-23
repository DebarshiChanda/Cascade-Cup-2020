# Cascade-Cup-2020

## APPROACH
1. Ensemble of Random Forest, XgBoost, CatBoost, LightGBM, KNN
2. Stratified KFold with the same seed used for all the models
3. UserId, Unnamed: 0 increased accuracy and F1_score
4. Creations == 0 perfectly classified class 1
5. Non zero creations had all the classes equally distributed
6. Trained XgBoost on the dataset with only non-zero creations and Hardcoded class to be 1 whenever creations are 0
7. Trained XgBoost on the entire dataset with a new Binary feature "zero creations"
8. Decomposed the feature columns to 3 using PCA
9. Trained XgBoost on the entire dataset by adding the 3 new decomposed columns
10. Feature selection using feature importances of the models and Recursive Feature Elimination CV
11. Out-of-fold cross-validation(OOF) was used instead of average across all the folds for deciding weights of the ensemble. [Source](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175614)

## MODELS AND NOTEBOOKS
1. RANDOM FOREST - 1 
  - Important Features were selected using feature importances.
  - Two Random Forests were combined and used together for better predictions.
  - The soft probabilities from both the trees were averaged to get the final probabilities.
  - [Notebook](https://www.kaggle.com/yerramvarun/randomforest) 

2. RANDOM FOREST - 2
  - Important Features were selected using Recursive Feature elimination.
  - [Recursive Feature Elimination Notebook](https://www.kaggle.com/debarshichanda/rfe-cv)
  - User Id and Unnamed used for better age_group prediction.
  - [Notebook](https://www.kaggle.com/debarshichanda/random-forest-with-unnamed)

3. LIGHT GBM
  - User Id and Unnamed used for better age_group prediction.
  - Tuned the parameters for the Number of leaves with regularization to increase the score.
  - Explored the double trees (class provided in the notebook), but the score decreased.
  - [Notebook](https://www.kaggle.com/yerramvarun/lightgbm)

4. CATBOOST
  - User Id and Unnamed used for better age_group prediction.
  - The parameters were tuned using Sklearn optimizer and the tuning code is provided.
  - Triple Tree was explored but the score decreased with the addition of User Id.
  - [Notebook](https://www.kaggle.com/yerramvarun/catboost-tuned)

5. KNN CLASSIFIER
  - Trained KNN Classifier on GPU using RAPIDS Library.
  - Found the best number of neighbors using the Elbow method.
  - [Notebook](https://www.kaggle.com/yerramvarun/knn-classifier)

6. DOUBLE XGBOOST - WITH MANUAL TUNING
  - Removed Features using feature importances and Recursive feature elimination.
  - Used Two Xgboosts together for better predictions.
  - The Parameters were tuned manually.
  - [Notebook](https://www.kaggle.com/debarshichanda/double-xgboost-manual-tuning)

7. XGBOOST - BASELINE
  - Removed features using Recursive Feature elimination.
  - Trained Baseline Xgboost with no parameter tuning.
  - [Notebook](https://www.kaggle.com/yerramvarun/double-xgboost)

8. XGBOOST - UNNAMED
  - Removed features using feature importances.
  - Used Unnamed: 0 and user id feature for better classification.
  - Tuned the number of trees and regularization parameters by hand and the results are documented in the form of comments.
  - [Notebook](https://www.kaggle.com/yerramvarun/xgboost-withunnamed)

9. XGBOOST - UNNAMED AND NON ZERO TRAINING
  - Removed features using feature importances.
  - Trained only on the samples with Creations value non zero which resulted in faster training and better results.
  - The Samples which had Creations zero were hardcoded and classified to age group 1.
  - The above method was discovered after the EDA of the training data.
  - [Notebook](https://www.kaggle.com/yerramvarun/xgboost-withunnamed-nonzerotraining)

10. XGBOOST - UNNAMED AND BINARY FEATURES
  - Removed Features using feature importances and Recursive feature elimination.
  - Created New Binary features with Creation feature which gave the model the information about the sparse nature of the Creation column in the training data.
  - [Notebook](https://www.kaggle.com/debarshichanda/xgboost-withunnamed-deb)

11. XGBOOST - PCA
  - Used Principal Component Analysis (PCA) for decomposing the data and creating new features for the model.
  - The PCA Features were used in addition to the original data so as to not lose the information from the old data.
  - [Notebook](https://www.kaggle.com/yerramvarun/xgboost-pca)

12. ENSEMBLING 
  - All the above models were used to generate OOF Files which were then blended using scipy optimizer.
  - We found appropriate weights using the oof files and used them to combine predictions.
  - The Blending increased the f1 score as much as by 1.2 
  - [Notebook](https://www.kaggle.com/yerramvarun/model-ensembling/data?scriptVersionId=49096504)

## WHAT DIDN’T WORK
1. Neural Networks couldn't cross 70. We tried both normal Neural Network and a Skip connection (Resnet) Model. [(Approach)](https://www.kaggle.com/yerramvarun/neural-network-cascade-cup?scriptVersionId=49406227)
2. Tabnet couldn't cross 67, fluctuated a lot
3. Training on GPU resulted in lower f1_score (0.3 less than CPU). [(RESOURCE)](https://github.com/dmlc/xgboost/issues/2300)
4. T-SNE decomposition very slow (9 hours on GPU not enough)
5. Tried using kernelPCA but faced a memory limit error.
6. SVM very slow (9 hours on CPU not enough for completing even a single fold)
7. Kaggle results not reproducible on Colab
8. Fast.ai Tabular learner gave high training loss

## What more could be done
1. Better Hyperparameter Search using Optuna [(our approach)](https://www.kaggle.com/debarshichanda/xgboost-tuning)
2. Better Feature engineering can be done.
3. More Deep learning approaches can be explored.

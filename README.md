# SC1015-Diabetes_Prediction

## About

This repository contains our mini-project for SC1015 (Introduction to Data Science & Artifical Intelligence). Our project is about identifying important health metrics in predicting the risk of diabetes.

<strong>Contributors</strong>
* @lohzhishen
* @YoNG-Zaii
* @TANERNHONG

<strong>Presentation</strong>

* Click on the picture below to be directed to the Youtube video. Alternatively, here is the Youtube video link: https://www.youtube.com/watch?v=0OaJ4Ntqq3I

[![Diabetes Prediction Youtube Video](https://lh3.googleusercontent.com/d/1tB22n_bbjxmbFUJUb-rvc9ckFaXJU6Lg)](https://www.youtube.com/watch?v=0OaJ4Ntqq3I "Diabetes Prediction Video Presentation")

<strong>Usage</strong>

Download "heart_2020_cleaned.csv" and "SC1015_Intro_to_DSAI.ipynb". Place both files in the same folder.

Jupyter Notebook:

* Run normally.

Google Colaboratory:

* Upload "SC1015_Intro_to_DSAI.ipynb" to Google Colab.
* Upload "heart_2020_cleaned.csv" to runtime environment (Ensure that the whole file is uploaded before running the notebook or there will be errors.)

## Problem Statement

> <em>What are some of the important health metrics in determining the risk of diabetes?</em>

<strong>Approach</strong>

Our group approach to answering this question through a data-driven method is to reframe the problem as a <strong>classification</strong> problem. From the models that we have trained, we extracted the relative feature importance and used this information to come to a conclusion about the importance of the different health metrics.

## Dataset used

> <em>Dataset from Kaggle: "Personal Key Indicators of Heart Disease" by Kamil Pytlak</em> <br>
> <em>Source : https://www.kaggle.com/kamilpytlak/personal-key-indicators-of-heart-disease (requires login to download)</em>

Although we are not using the dataset for its intended purpose, the dataset does provide the essential information we want, mainly whether an individual has diabetes and their health metrics.

## New Tools Used

<strong>EDA:</strong>

* Chi-square test of independence (chi2_contingency from scipy), Cramer's V and Tsuchuprow's T.

<strong>Data preprocessing:</strong>

* Under sampling majority class (RandomUnderSampler from imblearn)
* Class weights
* Transforms (StandardScaler and PolynomialFeatures from sklearn)

<strong>Models:</strong>

* Logistic regression (LogisticRegression from sklearn)
* Random forest classifier (RandomForestClassifier from sklearn)

<strong>Evaluation tools:</strong>

* Evaluation reports (classification_report from sklearn)
* ROC curve and AUC score (roc_curve and roc_auc_score from sklearn)
* Recursive feature elimination (RFE from sklearn) 

## Conclusions

<strong>Results</strong>

<em>Accuracy of models</em>

The LogisticRegression is the more accurate model as it outperforms the RandomForestClassifier in terms of accuracy and has a higher AUC score. It also has a lower false positive and false negative rate.

<em>Feature importance</em>

The LogisticRegression model placed approximately equal importance to BMI, AgeCategory and GenHealth as predictors for diabetes. This is in contrast to the RandomForestClassifier. It placed the high importance on BMI. However, GenHealth and AgeCategory are about half as important as BMI, and SleepTime is about a quarter as important as BMI.

<strong>Insights</strong>

Both models suggest that BMI is an important factor and SleepTime is a relatively unimportant factor in predicting the risk of diabetes.

However, LogisticRegression placed importance on the variables more evenly, with SleepTime being an aforementioned exception, whereas the RandomForestClassifier placed greater relative feature importance on BMI as compared to the rest of the variables.

* <strong>BMI, Age and General Health are important health metrics in predicting the risk of diabetes.</strong>
* <strong>Sleep Time is a relative unimportant health metric in predicting the risk of diabetes.</strong>

## Learning Points

* Handling imbalanced datasets using resampling methods and class weights.
* Parametric and nonparametric machine-learning algorithms - Logistic regression and RandomForestClassifier respectively - from sklearn.
* Recursive feature elimination and sklearn package.
* Collaborating using Google Colab.
* Concepts of PolynomialTransform and StandardScaler.
* Concepts of Chi-square Test of Independence, Cramer's V, and Tsuchuprow's T.

## References

Bali A. (2022, April 5). <em>Chi-Square Formula: Definition, P-value, Applications, Examples</em>. collegedunia. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://collegedunia.com/exams/chi-square-formula-definition-pvalue-applications-examples-articleid-4167#fi  

Chao, D. Y. (2021, May 22). <em>Chi-Square Test, with Python</em>. Towards Data Science. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://towardsdatascience.com/chi-square-test-with-python-d8ba98117626  

<em>Common pitfalls and recommended practices</em>. (n.d.). Scikit-learn. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://scikit-learn.org/stable/common_pitfalls.html  

<em>Diabetes.</em> (2022). World Health Organization.<br>&nbsp;&nbsp;&nbsp;&nbsp;
https://www.who.int/health-topics/diabetes#tab=tab_1

<em>Imbalanced Data</em>. (2011, Nov 1). Google Developers. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data  

Pytlak, K. (2022). <em>Personal Key Indicators of Heart Disease</em>. Kaggle. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

<em>RandomUnderSampler</em>. (n.d.). ImbalancedLearn. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html

Richmond, S. (2016, March 21). <em>Algorithms Exposed: Random Forest</em>. bccvl. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://bccvl.org.au/algorithms-exposed-random-forest/#:~:text=ASSUMPTIONS,are%20ordinal%20or%20non%2Dordinal  

Sarang, N. (2018, June 27). <em>Understanding AUC - ROC Curve</em>. Towards Data Science. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

Seb. (2021, April 8). <em>Chi-Square Distribution Table</em>. Programmathically. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://programmathically.com/chi-square-distribution-table/  

<em>Singapore's War on Diabetes</em>. (2021, May 26). HealthHub. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://www.healthhub.sg/live-healthy/1273/d-day-for-diabetes

<em>sklearn.ensemble.RandomForestClassifier</em>. (n.d.). Scikit-learn. <br>&nbsp;&nbsp;&nbsp;&nbsp; 
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

<em>sklearn.feature_selection.RFE</em>. (n.d.). Scikit-learn. <br>&nbsp;&nbsp;&nbsp;&nbsp; 
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html?highlight=rfe#sklearn.feature_selection.RFE

<em>sklearn.linear_model.LogisticRegression</em>. (n.d.). Scikit-learn. <br>&nbsp;&nbsp;&nbsp;&nbsp; 
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

<em>sklearn.metrics.classification_report</em>. (n.d.). Scikit-learn. <br>&nbsp;&nbsp;&nbsp;&nbsp; 
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report

<em>sklearn.metrics.roc_auc_score</em>. (n.d.). Scikit-learn. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score

<em>sklearn.metrics.roc_curve</em>. (n.d.). Scikit-learn. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve  

<em>sklearn.preprocessing.PolynomialFeatures</em> (n.d.). Scikit-learn. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures

<em>sklearn.preprocessing.StandardScaler</em>. (n.d.). Scikit-learn. <br>&nbsp;&nbsp;&nbsp;&nbsp;
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler 

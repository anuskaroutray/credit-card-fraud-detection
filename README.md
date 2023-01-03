# Credit Card Fraud Detection

Nowadays credit card misuse is more likely as a result of increased credit card use. Both the owners of credit cards and financial institutions suffer large financial losses as a result of credit card theft. The primary goal of this paper was to identify such scams, which includes public data accessibility and high-class unbalanced data, evolving fraud tactics, and high false alert rates. There are numerous machine learning-based methods for detecting credit cards, including the Decision Tree, Random Forest, Support Vector Machine, Logistic Regression, and XG Boost. However, because of their poor accuracy, modern deep learning algorithms are used. To get effective results, a comparative examination of both machine learning and deep learning algorithms was done. The European card benchmark dataset was used for fraud detection. The dataset was first subjected to a machine learning technique, which somewhat increased the accuracy of fraud detection.Later, three convolutional neural network-based designs were used to boost the effectiveness of fraud detection. The precision of detection was further improved by adding more layers. By varying the number of hidden layers, epochs, and using the most recent models, a thorough empirical investigation was conducted. Enhanced outcomes were obtained, with optimum values for accuracy, precision, recall, and f1-score of 98.5%, 97.28%, 86.6%, and 91.63%, respectively. For situations involving credit card detection, the suggested model performs better than cutting-edge machine learning and deep learning techniques.

## Dataset Decription

For research purposes, the credit card dataset is accessible. The dataset contains transactions that a cardholder completed over the course of two days in September 2018. In total, there were 284,807 transactions, and 492, or 0.172 percent, of those were fraudulent. Principal component analysis is used to apply the main component analysis to the majority of the dataset's features since exposing a consumer's transactional information is regarded as a concern of confidentiality (PCA). In the relevant literature, PCA is a common and commonly used method for lowering the dimensionality of such datasets, improving interpretability while minimising information loss. It accomplishes this by producing fresh, uncorrelated variables that maximise variance one after the other. There are a total of 28 variables, along with the time step and the amount for the transaction.

## Exploratory Data Analysis

1. Visualise Class Distribution of the dataset
2. Visualise several statistical quantities of the dataset

## Methods

After thorough exploratory data analysis, we leverage several machine learning and deep learning algorithms. The list of the algorithms are mentioned below.

- Machine Learning Algorithms
  1. Decision Tree Classifier
  2. K Nearest Neighbours
  3. Support Vector Machines
  4. Random Forest
  5. Logistic Regression 
  6. XGBoost
- Deep Learning Algorithms
  1. Baseline CNN with 11 layers
  2. CNN with 14 layers
  3. CNN with 17 layers
  4. CNN with 20 layers
  
We also introduced data balancing using SMOTE oversampling, which showed comprehensive increase in the evaluation metrics (accuracy, recall, precision, f1-score)

## Conclusion

In this paper, authors claim an efficient approach for credit card fraud detection. They use CNN to build the credit card detection pipeline and employ oversampling using SMOTE to tackle data imbalance. The authors claim that proposed methodology outperforms other approaches using SVM, Decision Tree, XGBoost when compared using accuracy, precision, f1 etc. In order to validate their claims, they carry out exhaustive experiments and according to our validation, the claims are indeed true.

# Credit Card Fraud Detection: Project Overview

_It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. This is a machine learning project that uses some anomaly detection algorithms in order to classify fraud and normal transactions._

## General Information
* [Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) - The dataset was downloaded in the Kaggle Repository.
* Created a machine learning models that is capable of detect whether a transaction is normal or fraudulent,by analyzing different transaction features, the time between transactions, the amount of money, among others.
* Created a Deep Neural Network to classify the credit card transactions, getting an accuracy of 90 %.
* Used some visualization techniques to visualize the data. Also, by Exploratory Data Analysis, I found the most relevant features to classify transactions.
* Isolation Forest is the model with the best performance, with the highest accuracy
* Isolation Forest detected 57 errors versus Local Outlier Factor detecting 97 errors vs. SVM detecting 1420 errors
* When comparing error precision & recall for the 3 models , the Isolation Forest performed much better than the LOF and SVM with 42 % against 2% (LOF) and 3% (SVM).
* In order to improve the performance of the models you can use:
   * More complex models of anomaly detection
   * Use more samples of the fraud class, in this case there was a serious imbalance class problem and that is why the models are not the best.


## Resources
* **Python Version:** 3.7
* **Scraper GitHub:** https://github.com/krishnaik06/Credit-Card-Fraudlent
* **Packages:** Pandas, Numpy, Matplotlib, Seaborn and Sklearn
* **Programs:** Jupyter Notebook

## Exploratory Data Analysis (EDA)

First, I plotted the numbers of instances per class in order to know how to handle the data. In this case, I found a very serious imbalance class problem, whereas for normal transactions there were 284315 instances and for the fraud transactions there were only 492 instances.

![classes](https://user-images.githubusercontent.com/63115543/92504259-05b93b00-f1c8-11ea-87b0-e1b87b5c38e4.jpg)

Secondly, I plotted the frequency  of transactions throughout time for each class.

![fre](https://user-images.githubusercontent.com/63115543/92504538-6f394980-f1c8-11ea-8d86-741fc937d3b8.jpg)

And finally, I plotted the scatter of the amount of transactions throughout time for each class.

![amount](https://user-images.githubusercontent.com/63115543/92504766-b7f10280-f1c8-11ea-84d0-c38bbc83799c.jpg)

## Data Visualization

### TSNE
_t-distributed Stochastic Neighbor Embedding (t-SNE). Is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results._

* Class 0 - Normal Transactions
* Class 1 - Fraud Transactions

![tsne](https://user-images.githubusercontent.com/63115543/92504987-08686000-f1c9-11ea-8987-0f92b832490d.jpg)

## Using Deep Learning to Classify the Credit Card Transactions

_By using Convolutional Neural Network and Balancing the Classes using undersampling for the Fraud Class, I got a tetsing accuracy 0f 89 %._

The Training Curve of the model for the accuracy is the following

![cnnacc](https://user-images.githubusercontent.com/63115543/93691244-29be2b80-faa8-11ea-964a-24d245d37e75.jpg)

And the Confusion Matrix for the testing set, with 176 instances predicted correctly and only 21 incorrectly.

![cm](https://user-images.githubusercontent.com/63115543/93691261-5c682400-faa8-11ea-86ac-36cf16b3e273.jpg)

## Machine Learning Models Performance

The following are the classification reports for each model:

### Isolation Forest
* **Detected Errors:** 57
* **Precision Detecting Fraud Class:** 42%
* **Macro F1 Score:** 71%

### Local Outlier Factor
* **Detected Errors:** 97
* **Precision Detecting Fraud Class:** 2%
* **Macro F1 Score:** 51%

### Support Vector Machine
* **Detected Errors:** 1420
* **Precision Detecting Fraud Class:** 3%
* **Macro F1 Score:** 52%

As you can see, the CNN has a very good performance with a high accuracy value, and the Machine Learning model with the best performance is the Isolation Forest, in order to improve the classification problem you can use more complex anomaly detection models or more complex Neural Networks.



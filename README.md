# Linear-Regression-with-multiple-variable
Here Iam using Linear Regression on multiple variable datasets.


# Feature-Engineering//Handling Missing Values:
#### The goal of feature engineering is simply to make your data better suited to the problem at hand.
   In Data Science, the performance of the model is depending on data preprocessing and data handling. Suppose if we build a model without Handling data, we got an accuracy of around 70%. By applying the Feature engineering on the same model there is a chance to increase the performance from 70% to more.

  feature Engineering you will learn:
 - determine which features are the most important with mutual information
 - invent new features in several real-world problem domains
 - encode high-cardinality categoricals with a target encoding
 - create segmentation features with k-means clustering
 - decompose a dataset's variation into features with principal component analysis
     
     Feature Engineering is importent part in ML.In this process there are many STEPS.
  
  ## 1* Feature Selection
                  (ON ANOTHER REPOSITORY)
  ## 2* Handling missing values:
  
In real world data, there are some instances where a particular element is absent because of various reasons, such as, corrupt data, failure to load the information, or incomplete extraction. Handling the missing values is one of the greatest challenges faced by analysts, because making the right decision on how to handle it generates robust data models. Let us look at different ways of imputing the missing values.
     
![image](https://user-images.githubusercontent.com/83010684/142798630-9b918502-a85b-4138-bb4d-569e199da9cd.png)

  #### Replacing With Mean/Median/Mode:
  
This strategy can be applied on a feature which has numeric data like the age of a person or the ticket fare. We can calculate the mean, median or mode of the feature and replace it with the missing values. This is an approximation which can add variance to the data set. But the loss of the data can be negated by this method which yields better results compared to removal of rows and columns. Replacing with the above three approximations are a statistical approach of handling the missing values. This method is also called as leaking the data while training. Another way is to approximate it with the deviation of neighbouring values. This works better if the data is linear.  
  
        In the missing value places, to replace the missing values with mean or median to numerical data and for categorical data with mode.

  ![image](https://user-images.githubusercontent.com/83010684/142797825-0f8cac56-9a90-4729-b8ba-2b37847a69b5.png)
##### Pros:
This is a better approach when the data size is small
It can prevent data loss which results in removal of the rows and columns
##### Cons:
Imputing the approximations add variance and bias
Works poorly compared to other multiple-imputations method
  
 #### Deleting Rows:
  
This method commonly used to handle the null values. Here, we either delete a particular row if it has a null value for a particular feature and a particular column if it has more than 70-75% of missing values. This method is advised only when there are enough samples in the data set. One has to make sure that after we have deleted the data, there is no addition of bias. Removing the data will lead to loss of information which will not give the expected results while predicting the output.

- Drop NA values entire rows.
- ![image](https://user-images.githubusercontent.com/83010684/142797896-daa638fc-e279-4b44-afbb-30d49fc50612.png)

- Drop NA values entire features. (it helps if NA values are more than 50% in a feature)
 ![image](https://user-images.githubusercontent.com/83010684/142797966-a8213a36-44ab-4e07-a551-9bfe9a88c35f.png)

- Replace NA values with 0.
 
![image](https://user-images.githubusercontent.com/83010684/142797987-9077b003-9f76-47f7-97f8-68035e9f304d.png)


##### Pros:
Complete removal of data with missing values results in robust and highly accurate model
Deleting a particular row or a column with no specific information is better, since it does not have a high weightage
##### Cons:
Loss of information and data
Works poorly if the percentage of missing values is high (say 30%), compared to the whole dataset
       
#### Assigning An Unique Category

A categorical feature will have a definite number of possibilities, such as gender, for example. Since they have a definite number of classes, we can assign another class for the missing values. Here, the features Cabin and Embarked have missing values which can be replaced with a new category, say, U for ‘unknown’. This strategy will add more information into the dataset which will result in the change of variance. Since they are categorical, we need to find one hot encoding to convert it to a numeric form for the algorithm to understand it.

![image](https://user-images.githubusercontent.com/83010684/142799835-511f3a85-d4d5-4164-85e0-f2125d03de10.png)

 ##### Pros:
Less possibilities with one extra category, resulting in low variance after one hot encoding — since it is categorical
Negates the loss of data by adding an unique category
##### Cons:
Adds less variance
Adds another feature to the model while encoding, which may result in poor performance
#### Predicting The Missing Values

Using the features which do not have missing values, we can predict the nulls with the help of a machine learning algorithm. This method may result in better accuracy, unless a missing value is expected to have a very high variance. We will be using linear regression to replace the nulls in the feature ‘age’, using other available features. One can experiment with different algorithms and check which gives the best accuracy instead of sticking to a single algorithm.

![2021-11-22 (2)](https://user-images.githubusercontent.com/83010684/142806936-06367f33-947c-4c6f-8d6a-d4cdc0bad2fc.png)


##### Pros:
Imputing the missing variable is an improvement as long as the bias from the same is smaller than the omitted variable bias
Yields unbiased estimates of the model parameters
##### Cons:
Bias also arises when an incomplete conditioning set is used for a categorical variable
Considered only as a proxy for the true values

we can try an alternative CCA approach by creating a new field representing the missing values. In the example below, we created a new field called “Missing Age” that contains a 1 when the “Age” variable is null, otherwise, it contains a 0.
  
#### Using Algorithms Which Support Missing Values
Here can be used here is RandomForest. This model produces a robust result because it works well on non-linear and the categorical data. It adapts to the data structure taking into consideration of the high variance or the bias, producing better results on large datasets.
##### Pros:
Does not require creation of a predictive model for each attribute with missing data in the dataset
Correlation of the data is neglected
##### Cons:
Is a very time consuming process and it can be critical in data mining where large databases are being extracted
Choice of distance functions can be Euclidean, Manhattan etc. which is do not yield a robust result

#### End of Distribution Imputation
If there is suspicion that the missing value is not at random then capturing that information is important. In this scenario, one would want to replace missing data with values that are at the tails of the distribution of the variable. The advantage is that it is quick and captures the importance of missing values (if one suspects the missing data is valuable). On the flipside, performing this action may distort the variable, mask predictive power if missingness is not important, hide true outliers if the missing data is large or create an unintended outlier if N/As are small.  Once again, this method should be performed on the training set and propagated on the test set.  Since we know “Age” follows a normal distribution, outliers will be computed using the mean rather than the median.

![image](https://user-images.githubusercontent.com/83010684/142810027-00a56e8a-7cde-4da4-9b29-065be83b07b5.png)


  ## 3* Handling imbalanced data
                  (ON ANOTHER REPOSITORY)
  
  ## 4* Handling outliers
                  (ON ANOTHER REPOSITORY)
  
  ## 5* Binning
                  (ON ANOTHER REPOSITORY)
  
  ## 6* Encoding
                  (ON ANOTHER REPOSITORY)
  
  ## 7* Feature Scaling
                  (ON ANOTHER REPOSITORY)
  
  
  reference:https://www.analyticsvidhya.com/blog/2021/03/step-by-step-process-of-feature-engineering-for-machine-learning-algorithms-in-data-science/


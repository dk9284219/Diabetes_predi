# Diabetes_predi
this project to Diabetes predictio to Solving to prediction 
#CANDIDATE’S DECLARATION
I Deepak kumar hereby certify that the work, which is being presented in the Mini-project report, entitled Diabetes prediction using machine learning, in partial fulfillment of the requirement for the award of the Degree of  Masters in Computer Application and submitted to the institution is an authentic record of my own work carried out during the period Nov-2022 to March -2023 under the supervision of Dr. Manoj Kumar Singh. I also cited the reference about the text(s) /table(s) from where they have been taken.

The matter presented in this Mini-project has not been submitted elsewhere for the award of any other degree or diploma from any Institutions.






INTRODUCTION




All around there are numerous ceaseless infections that are boundless in evolved and developing nations. One of such sickness is diabetes. Diabetes is a metabolic issue that causes blood sugar by creating a significant measure of insulin in the human body or by producing a little measure of insulin. Diabetes is perhaps the deadliest sickness on the planet. It is not just a malady yet, also a maker of different sorts of sicknesses like a coronary failure, visual deficiency, kidney ailments and nerve harm, and so on. Subsequently, the identification of such chronic metabolic ailment at a beginning period could help specialists around the globe in forestalling loss of human life. Presently, with the ascent of machine learning, AI, and neural systems, and their application in various domains [1, 2] we may have the option to find an answer for this issue. ML strategies and neural systems help scientists to find new realities from existing well-being-related informational indexes, which may help in ailment supervision and detection. The current work is completed utilizing the Pima Indians Diabetes Database. The point of this framework is to make an ML model, which can anticipate with precision the likelihood or the odds of a patient being diabetic. The ordinary distinguishing process for the location of diabetes is that the patient needs to visit a symptomatic focus. One of the key issues of bio-informatics examination is to achieve precise outcomes from the information. Human mistakes or various laboratory tests can entangle the procedure of identification of the disease. This model can foresee whether the patient has diabetes or not, aiding specialists to ensure that the patient in need of clinical consideration can get it on schedule and also help anticipate the loss of human lives.
DNA makes neural networks the apparent choice. Neural networks use neurons to transmit data across various layers, with each node working on a different weighted parameter to help predict diabetes.
Presently, with the ascent of machine learning, AI, and neural systems, and their application in various domains [1, 2] we may have the option to find an answer for this issue. ML strategies and neural systems help scientists to find new realities from existing well-being-related informational indexes, which may help in ailment supervision and detection. The current work is completed utilizing the Pima Indians Diabetes Database.

Types of Diabetes 

2.1	Type 1

diabetes means that the immune system is compromised and the cells fail to produce insulin in sufficient amounts. There are no eloquent studies that prove the causes of type 1 diabetes and there are currently no known methods of prevention.
2.2	Type 2

diabetes means that the cells produce a low quantity of insulin or the body can’t use the insulin correctly. This is the most common type of diabetes, thus affecting 90% of persons diagnosed with diabetes. It is caused by both genetic factors and the manner of living. 

2.2	Gestational

diabetes appears in pregnant women who suddenly develop high blood sugar. In two thirds of the cases, it will reappear during subsequent pregnancies. There is a great chance that type 1 or type 2 diabetes will occur after a pregnancy affected by gestational diabetes

Symptoms of Diabetes
• Frequent Urination 
• Increased thirst 
• Tired/Sleepiness 
• Weight loss 
• Blurred vision 
• Mood swings 
• Confusion and difficulty concentrating 
• frequent infections 

Causes of Diabetes  

Genetic factors are the main cause of diabetes. It is caused by at least two mutant genes in the chromosome 6, the chromosome that affects the response of the body to various antigens. Viral infection may also influence the occurrence of type 1 and type 2 diabetes. Studies have shown that infection with viruses such as rubella, Coxsackievirus, mumps, hepatitis B virus, and cytomegalovirus increase the risk of developing diabetes.









Related Works


Diabetes prediction is a classification technique with two mutually exclusive possibleoutcomes, either the person is diabetic or not diabetic. After extensive research, wecame to conclusion that although numerous classification techniques can be used forthe purpose of prediction, the observed accuracy varied. On careful examination of the performance of techniques used in prevalent works, logistic regression, KNN,Naive Bayes[3], random forest, , and neural network [4], we found themat par when applied to our dataset. KNN and logistic regression techniques were ableto achieve 96% accuracy.




The primary factor which influenced our algorithm selection was its adaptabilityand compatibility with future applications. The inevitable shift of data storage toward DNA makes neural networks the apparent choice. Neural networks use neurons totransmit data across various layers, with each node working on a different weightedparameter to help predict diabetes.




The point of this framework is to make an ML model, which can anticipatewith precision the likelihood or the odds of a patient being diabetic. The ordinarydistinguishing process for the location of diabetes is that the patient needs to visit asymptomatic focus. One of the key issues of bio-informatics examination is to achieveprecise outcomes from the information. Human mistakes or various laboratory testscan entangle the procedure of identification of the disease. This model can foreseewhether the patient has diabetes or not, aiding specialists to ensure that the patientin need of clinical consideration can get it on schedule and also help anticipate theloss of human lives








DATASET 

The dataset collected is originally from the Pima Indians Diabetes Database is available on Kaggle. It consists of several medical analyst variables and one target variable. The objective of the dataset is to predict whether the patient has diabetes or not. The dataset consists of several independent variables and one dependent variable, i.e., the outcome. Independent variables include the number of pregnancies the patient has had their BMI, insulin level, age, and so on as Shown in Following Table 1


Dataset Description

The diabetes data set was originated from https://www.kaggle.com/johndasilva/diabetes. Diabetes dataset containing 2000 cases. The objective is to predict based on the measures to predict if the patient is diabetic or not


➔ The diabetes data set consists of 2000 data points, with 9 features each. 
➔ “Outcome” is the feature we are going to predict, 0 means No diabetes, 1 means diabetes.




➔ There is no null values in dataset. 























 Correlation Matrix:  






It is easy to see that there is no single feature that has a very high correlation with our outcome value. Some of the features have a negative correlation with the outcome value and some have positive.
Histogram:  
Let’s take a look at the plots. It shows how each feature and label is distributed along different ranges, which further confirms the need for scaling. Next, wherever you see discrete bars, it basically means that each of these is actually a categorical variable. We will need to handle these categorical variables before applying Machine Learning. Our outcome labels have two classes, 0 for no disease and 1 for disease.






Bar Plot For Outcome Class 




The above graph shows that the data is biased towards datapoints having outcome value as 0 where it means that diabetes was not present actually. The number of non-diabetics is almost twice the number of diabetic patients.













PROPOSED METHODS
I] Dataset collection –
 It includes data collection and understanding the data to study the hidden patterns and trends which helps to predict and evaluating the results. Dataset carries 1405 rows i.e., total number of data and 10 columns i.e., total number of features. Features include Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, Age


II] Data Pre-processing:
This phase of model handles inconsistent data in order to get more accurate and precise results like in this dataset Id is inconsistent so we dropped the feature. This dataset doesn’t contain missing values. So, we imputed missing values for few selected attributes like Glucose level, Blood Pressure, Skin Thickness, BMI and Age because these attributes cannot have values zero. Then data was scaled using StandardScaler. Since therewere a smaller number of features and important for prediction so no feature selection was Done.


III]Missing value identification:     
Using the Panda library and SK-learn , we got the missing values in the datasets, shown in Table 2. We replaced the missing value with the corresponding mean value.









IV] Feature selection:
Pearson’s correlation method is a popular method to find the most vrelevant attributes/features. The correlation coefficient is calculated in this method, which correlates with the output and input attributes. The coefficient value remains in the range by between −1 and 1. The value above 0.5 and below −0.5 indicates a notable correlation, and the zero value means no correlation

V] Scaling and Normalization:


We performed feature scaling by normalizing the data from 0 to 1 range, which boosted the algorithm’s calculation speed. scaling means that you're transforming your data so that it fits within a specific scale, like 0-100 or 0-1. You want to scale data when you're using methods based on measures of how far a part data points are, like support vector machines (SVM) or k-nearest neighbours (KNN).With these algorithms, a change of "1" in any numeric feature is given the same importance

VI] Splitting of data:


After data cleaning and pre-processing, the dataset becomes ready to train and test. In the train/split method, we split the dataset randomly into the training and testing set. For Training we took 1600 sample and for testing we took 400 sample.

VII] Design and implementation of classification model:


In this research work, comprehensive studies are done by applying different ML classification techniques like KNN, RF, NB, LR, SVM.




VIII] Machine learning classifier:
We have developed a model using Machine learning Technique. Used different classifier and ensemble techniques to predict diabetes dataset. We have applied SVM, LR, DT and RF Machine learning classifier to analyse the performance by finding accuracy of each classifier All the classifiers are implemented using scikit learn libraries in python. The implemented classification algorithms are described in next section




















































MODELING AND ANALYSIS:


A] Logistic Regression:
Logistic regression is a machine learning technique used when dependent variables are able to categorize. The outputs obtained by using the logistic regression is based on the available features. Here sigmoidal function is used to categorize the output.

B] K-Nearest Neighbors:
K-nearest neighbors (KNN) algorithm uses ‘feature similarity’ to predict the values of new datapoints which further means that the new data point will be assigned a value based on how closely it matches the points in the training set. Predictions are made for a new instance (x) by searching through the entire training set for the K most similar instances (the neighbors) and summarizing the output variable for those K instances.

C]SVM:
SVM is supervised learning algorithm used for classification. In SVM we have to identify the right hyper plane to classify the data correctly. In this we have to set correct parameters values. To find the right hyper plane we have to find right margin for this we have choose the gamma value as 0.0001 and rbf kernel. If we select the hyper plane with low margin leads to miss classification.


F] Random Forest:
Random forest is an ensemble learning method for classification. This algorithm consists of trees and the number of tree structures present in the data is used to predict the accuracy.Where leaves are corresponds to the class labels and attributes are corresponds to internal node of the tree. Here number of trees in forest used is 100 in number and Gini index is used for splitting the nodes.



















Measurements


To find the effienct classifier for diabetes prediction we have applied a performance matrices are confusion matrix and accuracy are discussed as follows:
Confusion matrix: - which provides output matrix with complete description performance of the model.
Here, Tp: True positive
FP: False positive
TN: True negative
FN: False negative
The following performance metrics are used to calculate the presentation of various algorithms.
True positive (TP) – person has disease, and the prediction also has a positive
True negative (TN) – person not having disease and the prediction also has a negative
False positive (FP) – person not having disease but the prediction has a positive
False negative (FN) – person having disease and the prediction also has a positive
TP and TN can be used to calculate accuracy rate and the error rates can be computed using FP and FN values.
True positive rate can be calculated as TP by a total number of persons have disease in reality.
False positive rate can be calculated as FP by a total number of persons do not have
disease in reality.
Precision is TP/ total number of person have prediction result is yes.
Accuracy is the total number of correctly classified records


Accuracy- We have chooses accuracy matrix to measure the performance of all the models.The ratio of number of correct predictions to the total number of predictions Made









RESULTS AND DISCUSSION


Machine learning classification algorithms developed for prediction of diabetes in earlier stage. We used 80% of data for training and 20% of data for testing. In this ratio of data splitting Here we found that k-nearest neighbors Classifier predicted with 96% of accuracy as for the dataset. Comparison of results of all the implemented classifiers are listed below.

Algorithm
Accuracy
Random Forest
92.75%
SVM
75.93%
K-Nearest Neighbors
96.81%
Logistic Regression
76.06%
















Conclusion


The objective of the project was to develop a model which could identify patients with diabetes who are at high risk of hospital admission. Prediction of risk of hospital admission is a fairly complex task. Many factors influence this process and the outcome. There is presently a serious need for methods that can increase healthcare institution’s understanding of what is important in predicting the hospital admission risk. This project is a small contribution to the present existing methods of diabetes detection by proposing a system that can be used as an assistive tool in identifying the patients at greater risk of being diabetic. This project achieves this by analyzing many key factors like the patient’s blood glucose level, body mass index, etc., using various machine learning models and through retrospective analysis of patients’ medical records. The project predicts the onset of diabetes in a person based on the relevant medical details that are collected using aWeb application.When the user enters all the relevant medical data required in the online Web application, this data is then passed on to the trained model for it to make predictions whether the person is diabetic or nondiabetic. The model is developed using an artificial neural network consisting of a total of six dense layers. Each of these layers is responsible for the efficient working of the model. The model makes the prediction with an accuracy of 96% ,which is fairly good and reliable.




























References


1. Sahoo, K.S., et al.: An evolutionary SVM model for DDOS attack detection in software definednetworks. IEEE Access 8, 132502–132513 (2020)


2. Sahoo, K.S., et al.: A machine learning approach for predicting DDoS traffic in
software defined networks. In: 2018 International Conference on Information
Technology (ICIT). IEEE (2018)


3. Jakka, A., Vakula Rani, J.: Performance evaluation of machine learning models for diabetesprediction. Int. J. Innov. Technol. Explor. Eng. (IJITEE) 8(11) (2019). ISSN:2278-3075


4. Zou, Q., Qu, K., Luo, Y., Yin, D., Ju, Y., Tang, H.: Predicting diabetes mellitus with machine learning techniques. Bioinform. Comput. Biol. Sect. J. Front. Genet.,
published: 06 2018





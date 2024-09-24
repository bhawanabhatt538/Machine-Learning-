# Random Under-sampling and Random Over-sampling
# 1 Method
c_class_0, c_class_1 = data.Class.value_counts()
data_class_0 = data[data['Class']==0]
data_class_1 = data[data['Class']==1]

# 1. Undersampling 
data_class_0_under = data_class_0.sample(c_class_1)
data_test_under = pd.concat([data_class_0_under, data_class_1], axis=0)
# Concatening the data one after the other to create one single dataframe containing all the data
data_test_under.Class.value_counts().plot(kind='bar',title='Under Sampling')

#2. Oversampling
data_class_1_over = data_class_1.sample(c_class_0, replace=True)
# Here we are increasing number of records for label 1 by randomly generating data
data_test_over = pd.concat([data_class_0, data_class_1_over],axis=0)
print("After Random Over-sampling")
print(data_test_over.Class.value_counts())

data_test_over.Class.value_counts().plot(kind='bar',title='Random Over-Sampling')
# /*=========================================================================================================================================*/
# Random Under-sampling and Random Over-sampling
# 2 Method

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)# define oversampling strategy
X_over, y_over = ros.fit_resample(X_train, y_train)# fit and apply the transform
print('Genuine:', y_over.value_counts()[0], '/', round(y_over.value_counts()[0]/len(y_over) * 100,2), '% of the dataset')
print('Frauds:', y_over.value_counts()[1], '/',round(y_over.value_counts()[1]/len(y_over) * 100,2), '% of the dataset')

from imblearn.under_sampling import RandomUnderSampler# define undersampling strategy
rus = RandomUnderSampler(random_state=42)# fit and apply the transform
X_under, y_under = rus.fit_resample(X_train, y_train)
print('Genuine:', y_under.value_counts()[0], '/', round(y_under.value_counts()[0]/len(y_under) * 100,2), '% of the dataset')
print('Frauds:', y_under.value_counts()[1], '/',round(y_under.value_counts()[1]/len(y_under) * 100,2), '% of the dataset')


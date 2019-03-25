import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB


health_data = pd.read_csv('train_2v.csv')
test_data = pd.read_csv('test_2v.csv')
health_data = health_data.dropna()
test_data = test_data.dropna()

X = health_data.drop(columns=['id', 'heart_disease', 'work_type', 'smoking_status', 'stroke'])
test_x = test_data.drop(columns=['id', 'heart_disease', 'work_type', 'smoking_status'])
y = health_data.heart_disease
test_y = test_data.heart_disease

gender_dict = {'Male' : 0, 'Female' : 1, 'Other' : 2}
married_dict = {'No' : 0, 'Yes' : 1}
residence_type_dict = {'Rural' : 0, 'Urban' : 1}

X.gender = [gender_dict[item] for item in X.gender]
X.ever_married = [married_dict[item] for item in X.ever_married]
X.Residence_type = [residence_type_dict[item] for item in X.Residence_type]
test_x.gender = [gender_dict[item] for item in test_x.gender]
test_x.ever_married = [married_dict[item] for item in test_x.ever_married]
test_x.Residence_type = [residence_type_dict[item] for item in test_x.Residence_type]

hd_model = DecisionTreeClassifier()
hd_model.fit(X, y)

predictions =  hd_model.predict(test_x)
print(mean_absolute_error(test_y, predictions))


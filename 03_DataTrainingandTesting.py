import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utilies import sk_util as sk
from datetime import datetime
sns.set_style('whitegrid')

# Load and preprocess the signal data
signal_data = pd.read_csv('Datasets/Datalabelling.csv')
# ... preprocess signal_data as needed ...
sns.pairplot(signal_data, hue='labels', vars=['GyroX_L_Filtered', 'GyroY_L_Filtered', 'GyroZ_L_Filtered'])

# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(10, 8))
plt.title('Correlation Between The Variables')
sns.heatmap(signal_data.corr(), annot=True)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

X = signal_data.drop('labels', axis=1)
y = signal_data.labels
print(f"'X' shape: {X.shape}")
print(f"'y' shape: {y.shape}")
pipeline = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('std_scaler', StandardScaler())
])


# 1. Linear Kernel SVM
start_time1 = datetime.now()
print("1. Linear Kernel SVM")
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2)
model1 = LinearSVC(loss='hinge', dual=True)
model1.fit(X_train1, y_train1)
sk.sk_print_score(model1, X_train1, y_train1, X_test1, y_test1, train=True)
sk.sk_print_score(model1, X_train1, y_train1, X_test1, y_test1, train=False)
end_time1 = datetime.now()
print('Duration: {}\n'.format(end_time1 - start_time1))


# 2. Polynomial Kernel SVM
# This code trains an SVM classifier using a 2nd-degree polynomial kernel.
# The hyperparameter coef0 controls how much the model is influenced by high degree ploynomials
print("2. Polynomial Kernel SVM")
start_time2 = datetime.now()
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)
model2 = SVC(kernel='poly', degree=2, coef0=1)
model2.fit(X_train2, y_train2)
sk.sk_print_score(model2, X_train2, y_train2, X_test2, y_test2, train=True)
sk.sk_print_score(model2, X_train2, y_train2, X_test2, y_test2, train=False)
end_time2 = datetime.now()
print('Duration: {}\n'.format(end_time2 - start_time2))

# # 3. Radial Kernel SVM
# # Just like the polynomial features method, the similarity features can be useful with any
# from sklearn.svm import SVC
# from sklearn import svm
# start_time3 = datetime.now()
# X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.2)
# model3 = SVC(kernel='rbf', gamma=0.5, C=0.1)
# model3.fit(X_train3, y_train3)
# print("3. Radial Kernel SVM")
# sk.sk_print_score(model3, X_train3, y_train3, X_test3, y_test3, train=True)
# sk.sk_print_score(model3, X_train3, y_train3, X_test3, y_test3, train=False)
# end_time3 = datetime.now()
# print('Duration: {}\n'.format(end_time3 - start_time3))

# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# scaler = StandardScaler()
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# plt.figure(figsize=(8,6))
# plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap='plasma')
# plt.xlabel('First principal component')
# plt.ylabel('Second Principal Component')
# plt.show()


# Random Forest classifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Random Forest (RF) Model")
start_time4 = datetime.now()
# Split data into training and testing sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2)

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_rf, y_train_rf)
sk.sk_print_score(rf, X_train_rf, y_train_rf, X_test_rf, y_test_rf, train=True)
sk.sk_print_score(rf, X_train_rf, y_train_rf, X_test_rf, y_test_rf, train=False)
end_time4 = datetime.now()
print('Duration: {}\n'.format(end_time4 - start_time4))


# K-Nearest Neighbors Algorithm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print("K-Nearest Neighbors Algorithm (KNN model)")
start_time5 = datetime.now()
# Split the dataset into training and testing sets
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.2)
# Scale the features using StandardScaler:
scaler = StandardScaler()
X_train_knn = scaler.fit_transform(X_train_knn)
X_test_knn = scaler.transform(X_test_knn)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_knn, y_train_knn)
sk.sk_print_score(knn, X_train_knn, y_train_knn, X_test_knn, y_test_knn, train=True)
sk.sk_print_score(knn, X_train_knn, y_train_knn, X_test_knn, y_test_knn, train=False)
end_time5 = datetime.now()
print('Duration: {}\n'.format(end_time5 - start_time5))

# neural network (NN)


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- scikit-learn API --- 
# Q1

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
salary_predictions=np.array([4, 8]).reshape(-1, 1)

model = LinearRegression()                    
model.fit(years, salary)                
year_predication_4_vs_8 = model.predict(salary_predictions)          
print(f'''
    Predictions for 4yr vs 8yr respectively: {year_predication_4_vs_8}
    Slope: {model.coef_[0]}
    Intercept: {model.intercept_}
    ''')  

# Q2
x = np.array([10, 20, 30, 40, 50])
print(f"Shape of x array: {x.shape}")
x_matrix= x.reshape(-1, 1)
print(f"Shape of x matrix: {x_matrix.shape}")

#Array must be turned into 2d arrays not only because this is what the method expects but it makes each sample an object
# this reduces ambuity of what data is being analyzed by putting an 2d shape each element becomes the object that is being used as 
#an individual observation

# Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

Kmeans_model= KMeans(n_clusters=3, random_state=42)
Kmeans_model.fit(X_clusters)
labels=Kmeans_model.predict(X_clusters)
centroids=Kmeans_model.cluster_centers_
labels_per_centroid=np.bincount(labels)
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], label=labels, c=labels, cmap='viridis', s=60, alpha=0.7)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", c="black", s=1000,)
plt.title("Question 3 Set")

plt.tight_layout()
plt.savefig('assignments_02/outputs/kmeans_clusters.png')
# plt.show()

print(f'''
Cluster Centers: {centroids}
Label Count{labels_per_centroid}
''')

np.random.seed(42)
num_patients = 100
age= np.random.randint(20, 65, num_patients).astype(float)
smoker= np.random.randint(0, 2, num_patients).astype(float)
cost= 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# ---Linear Regression --- 

# Q1
plt.scatter(age, cost, c=smoker, cmap="coolwarm", s=60, alpha=0.7)
plt.title("Medical Cost vs Age")
plt.tight_layout()
plt.savefig('assignments_02/outputs/cost_vs_age.png')
# plt.show()


#There is a distinct gradient that seperates the smoker cost vs the non smokers. 
#Across all age the cost of medical care is higher for smokers compared to non smokers

# Q2
age_2d= age.reshape(-1, 1)
cost_2d= cost.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    age_2d, cost_2d, test_size=0.2, random_state=42
)
print(f'''
    X_train shape:{X_train.shape}
    X_test shape:{X_test.shape}
    y_train shape:{y_train.shape}
    y_test shape:{y_test.shape}
''')
# Q3
model = LinearRegression()                   
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)


print(f'''
    Slope: {model.coef_[0][0]}
    Intercept: {model.intercept_}
    RMSE: {np.sqrt(np.mean((y_pred - y_test) ** 2))}
    R^2: {model.score(X_test, y_test)}
''')

#The slope means that is the magnitude the cost increases based on age. So slope * age will get you the cost of medical care


# Q4

X_full = np.column_stack([age, smoker])

X_train, X_test, y_train, y_test = train_test_split(
    X_full, cost_2d, test_size=0.2, random_state=42
)
model_full = LinearRegression()                   
model_full.fit(X_train, y_train)  
y_pred = model_full.predict(X_test)

print(f'''
    Smoker x Age Model
    age coefficient: {model_full.coef_[0][0]}
    smoker coefficient: {model_full.coef_[0][1]}
    Intercept: {model_full.intercept_}
    RMSE: {np.sqrt(np.mean((y_pred - y_test) ** 2))}
    R^2: {model_full.score(X_test, y_test)}
''')

#adding the smoker

# Q5 -UNFINISHED
plt.scatter(X_full, cost, label=labels, c=labels, cmap='viridis', s=60, alpha=0.7)

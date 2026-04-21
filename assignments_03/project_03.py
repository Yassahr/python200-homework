import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import BytesIO
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
from sklearn.pipeline import Pipeline

  
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
#Task 1
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 
print('X Shape, Y Shape, Info X')
print(X.shape, y.shape, X.info())
print(f'% of spam = {y.sum()/len(y) *100}')
print(y.info())


key_features = ["word_freq_free", "char_freq_!", "capital_run_length_total"]

#Boxplots for key Features
for feature in key_features:
    spam_value= X.loc[y["Class"]==0, feature]
    ham_value= X.loc[y["Class"]==1, feature]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([ham_value, spam_value])
    ax.set_title(f"{feature} vs Spam Boxplots")
    ax.set_ylabel(feature)
    ax.set_xlabel("Spam or ham")
    plt.tight_layout()
    plt.savefig(f"assignments_03/outputs/spam_boxplots{feature}.png")
    # plt.show()
#the scale for each feature can vary alot, this can determine which type of logical regression makes more sense to implement.
#for example KNN is better datasets where scale between features isn't really large or really small but it there is variations 
#either best to choose a different preprocessing or do addition normalization and standardization
#these vary because for some features there is a drastic difference between spama and non spam

#task 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components = int(np.argmax(cumvar >= 0.90)) + 1

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(cumvar) + 1), cumvar * 100)
ax.set_xlabel("Principal Components")
ax.set_ylabel("Cumulative Explained Variance (%)")
ax.set_title("PCA v Variance")
plt.tight_layout()
fig.savefig("assignments_03/outputs/pca_cumulative_variance.png")
# plt.show()

print(f'n:{n_components}')
X_train_pca = pca.transform(X_train_scaled)[:, :n_components]
X_test_pca  = pca.transform(X_test_scaled)[:, :n_components]

#task 3

#knn
def stats(name, data, X_tr, y_tr, X_te, y_te):
    data.fit(X_tr, y_tr)
    y_pred = data.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print(f'''{name}\n
    Accuracy: {accuracy:.4f}
    classification:{classification_report(y_te, y_pred, target_names=["Ham", "Spam"])}
    ''')
    return data, y_pred, accuracy

knn_raw, _, _ = stats(
    "KNN Unscaled",
    KNeighborsClassifier(n_neighbors=5),
    X_train, y_train, X_test, y_test
)

knn_scaled, _, _ = stats(
    "KNN Scaled",
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled, y_train, X_test_scaled, y_test
)

knn_pca, _, _ = stats(
    f"KNN {n_components} components)",
    KNeighborsClassifier(n_neighbors=5),
    X_train_pca, y_train, X_test_pca, y_test
)

#decison tree
dt_results = {}
for depth in [3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc_dt = accuracy_score(y_train, dt.predict(X_train))
    test_acc_dt  = accuracy_score(y_test,  dt.predict(X_test))
    dt_results[depth] = (train_acc_dt, test_acc_dt)

print(dt_results)

#the best depth is 5 since the testing and the training data has the smallest gap

dt_optimal, _, _ = stats(
    f"Decision Tree",
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X_train, y_train, X_test, y_test
)

#RandomForestClassifier
rf, y_pred, _ = stats(
    "Random Forest Classifier",
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train, X_test, y_test
)
#logical regression
lr_scaled, _, _ = stats(
    "Logistic Regression - Scaled",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_scaled, y_train, X_test_scaled, y_test
)

lr_pca, _, _ = stats(
    f"Logistic Regression on PCA",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_pca, y_train, X_test_pca, y_test
)

#overall the Random tree performed the best with accuracy and precision ranging from 94-97%

#confusion matrix


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.savefig("assignments_03/outputs/best_model_confusion_matrix.png")

#false negative are alot mroe conmmon meaning emails being labeled as ham but they are spam
feature_names = list(X.columns)

# Decision tree importances
dt_top = pd.Series(dt_optimal.feature_importances_, index=feature_names)
print(f"Top 10 features – Decision Tree: {dt_top.nlargest(10).to_string()}")

# Random forest importances
rf_feature = pd.Series(rf.feature_importances_, index=feature_names)
print(f"Top 10 features – Random Forest: {rf_feature.nlargest(10).to_string()}")


top_rf = rf_feature.nlargest(15).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
top_rf.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 15 Feature Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("assignments_03/outputs/feature_importances.png", )
plt.close(fig)

#these models do not agree on the most important feature, It suggests that trees are better for analysis

#task 4

cv_models = [
    ("KNN (unscaled)", KNeighborsClassifier(n_neighbors=5), X_train),
    ("KNN (scaled)",   KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    (f"KNN (PCA {n_components})", KNeighborsClassifier(n_neighbors=5), X_train_pca),
    (f"Decision Tree (depth={5})",DecisionTreeClassifier(max_depth=5, random_state=42), X_train),
    ("Random Forest (100 trees)",    RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    ("Logistic Regression (scaled)", LogisticRegression(C=1.0, max_iter=1000,solver="liblinear"),X_train_scaled), 
    (f"Logistic Regression (PCA {n_components})",LogisticRegression(C=1.0, max_iter=1000,solver="liblinear"),X_train_pca)
]
cv_results={}
for name, data, X_cv in cv_models:
    scores = cross_val_score(data, X_cv, y_train, cv=5, scoring="accuracy")
    cv_results[name]=[float(scores.mean()), float(scores.std())]
print(cv_results)
#again the random tree has the highest mean accurac, Knn Scaled> KNN unscaled, and Devisoon tree has alot of varience

#task 5
# Pipeline 1
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf_pipe = rf_pipeline.predict(X_test)
print(f'''Random Forest 
    Pipeline accuracy: {rf_pipeline.score(X_test, y_test):.4f}")
    classification report: {classification_report(y_test, y_pred_rf_pipe, target_names=["Ham", "Spam"])}
    ''')

#pipeline 2
lr_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr_pipe = lr_pipeline.predict(X_test)
print(f'''  Non-Tree Logical Regression
    Pipeline accuracy: {lr_pipeline.score(X_test, y_test):.4f}")
    classification report: {classification_report(y_test, y_pred_lr_pipe,target_names=["Ham", "Spam"])}    
    ''')
#the pipeline was able to recreate the same results so it confirms that preprocessing was able to be encapsulated for streamlining ptocesses
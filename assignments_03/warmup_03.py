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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import seaborn as sns


iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


#--- Preprocessing ---
#Q1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print(f'''
    X_train: {X_train.shape}
    X_test: {X_test.shape}
    y_train: {y_train.shape}
    y_test: {y_test.shape}
''')

#Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled  = scaler.transform(X_test)   
x_trained_mean=X_train_scaled.mean(axis=1)
x_test_mean=X_test_scaled.mean(axis=1)
print(f'''
 Training data mean:{x_trained_mean}
 Test data mean:{x_test_mean}
''')

# Only training data, while test data fititng was omittred to ensure  validation bias is not present when testing model which would learn from the test data if it was fitted to it 

#--- KNN ---
#Q1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy for unscaled:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

#Q2

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

preds = knn.predict(X_test_scaled)

print("Accuracy for unscaled:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

#scaling makes the performance worse, this an anomaly but since the iris data is already pretty clean and scaling removed the natural seperation which lead to worse results

#Q3
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print(cv_scores)        
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")

#this score is alot more trustworthy than just a single score with K test/train since it is split inot 5 different groups and it is tested against all of them
#the variation is reduced and the average of all them result in a more stable model

#Q4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]  

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")

#k=7 has the mean that is closing to 1 and loweset standard deviation so I think this calues would be the best k value for this training set.

#--- Classifier Evaluation ---
#Q1
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=7)
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig('assignments_03/outputs/knn_confusion_matrix')
# plt.show()
#vericolor and the virginica are confused the most often

#--- The sklearn API: Decision Trees ---
dtc=DecisionTreeClassifier(max_depth=3, random_state=42)
dtc.fit(X_train, y_train)
preds = dtc.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print(f' Accurancy: {accuracy}')
classification_rep=classification_report(y_test, preds)
print(f' Classifcation report for Decisoion tree: {classification_rep}')
# the accurancy for the KNN with the k=7 and the decision tree were extremely close but the Decision tree over all has score that were more consistent though the correctness averages out to be similar
#I think scaled would results in better results since it focuses more making sense of class differences than how different each is 

#--- Logistic Regression and Regularization---
#Q1
scaler_full = StandardScaler()
X_train_scaled = scaler_full.fit_transform(X_train)
X_test_scaled = scaler_full.transform(X_test)

# ------------------------------------------------------------------------------------------------------
log_reg_full_01 = OneVsRestClassifier(LogisticRegression(C=0.01, max_iter=1000, solver="liblinear"))
log_reg_full_01.fit(X_train_scaled, y_train)

log_reg_full_1 = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
log_reg_full_1.fit(X_train_scaled, y_train)

log_reg_full_100 = OneVsRestClassifier(LogisticRegression(C=100, max_iter=1000, solver="liblinear"))
log_reg_full_100.fit(X_train_scaled, y_train)

def get_coef_df(model, feature_names):
    """Average absolute coefficients across all OvR binary classifiers."""
    coefs = np.mean([est.coef_[0] for est in model.estimators_], axis=0)
    return pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs
    })

coef_df_01  = get_coef_df(log_reg_full_01,  X.columns)
coef_df_1   = get_coef_df(log_reg_full_1,   X.columns)
coef_df_100 = get_coef_df(log_reg_full_100, X.columns)

for c_val, model, coef_df in [
    (0.01,  log_reg_full_01,  coef_df_01),
    (1.0,   log_reg_full_1,   coef_df_1),
    (100,   log_reg_full_100, coef_df_100),
]:
    print(f"""
    C value = {c_val}
    Total Size of all Coefficients: {np.abs(coef_df['coefficient']).sum():.4f}
    """)

coef_df_1["abs_importance"] = coef_df_1["coefficient"].abs()
coef_df_1.sort_values("abs_importance", ascending=False).head(10)


# ------------------------------------------------------------------------------------------------------
 
# --- PCA ---

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

print(f'''
    X Digits Shape: {X_digits.shape}
    Image Shapes: {images.shape}
''')

#Q1
fig, axes = plt.subplots(1, 10, figsize=(15, 2))

for i, ax in  enumerate(axes):
    index = np.where(y_digits == i)[0][0] 
    ax.imshow(images[index], cmap='gray_r')
    ax.set_title(str(i))
plt.savefig('assignments_03/outputs/sample_digits.png')
plt.show()


#Q2
pca = PCA( svd_solver="randomized", random_state=0)
pca.fit(X_digits)



scores = pca.transform(X_digits)
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  
plt.colorbar(scatter, label='Digit')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Digits')
plt.savefig('assignments_03/outputs/pca_2d_projection.png')
plt.show()

#there is some localization but alot of overlap with the digits

#Q3
cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(cumsum)
plt.xlabel('# of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Variance Explained')
plt.savefig('assignments_03/outputs/pca_variance_explained.png')
plt.show()
#between 10-15 components for 80% variance

#Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]
digits = list(range(5))

fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# Original row
for col, index in enumerate(digits):
    ax = axes[0, col]
    ax.imshow(images[index], cmap='gray_r')
    ax.set_title(f"{index}")
    ax.axis('off')

# Reconstruction rows
for row, n in enumerate(n_values, start=1):
    for col, index in enumerate(digits):
        ax = axes[row, col]
        recon = reconstruct_digit(index, scores, pca, n)
        ax.imshow(recon, cmap='gray_r')
        ax.set_title(f" n:{n}")
        ax.axis('off')

plt.suptitle('PCA Recons v Increasing n Components')
plt.savefig('assignments_03/outputs/pca_reconstructions.png')
plt.show()
# the vest recon number is when it is equal to 2 and 40
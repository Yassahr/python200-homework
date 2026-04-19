import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Task 1
file="assignments_02/student_performance_math.csv"

#using ; as the sepertator since semicolons are being used instead of the more commonly parsed comma
students_performance=pd.read_csv(file, sep=';')
print(students_performance.head(5))
students_performance.info()
print(f"Shape: {students_performance.shape}")

#histogram
plt.hist(students_performance["G3"], bins=21, color="red", edgecolor='black', density=True)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Quarter Grades")
plt.xlabel("Grade Freq")
plt.show()
plt.savefig('assignments_02/outputs/g3_distribution.png')


# Task 2

#data cleaning
#removed all of the null values for student who did not take the final exam
print(f'Shape before cleaning: {students_performance.shape}')
students_performance_unclean=students_performance.copy(deep=True)

students_performance['G3']=students_performance['G3'].replace(0, np.nan)
students_performance_filtered=students_performance.dropna(subset=['G3'])
print(f'Shape after removal: {students_performance_filtered.shape}')
students_performance_filtered.info()


#replacing yes/no with  binary
pd.set_option('future.no_silent_downcasting', True)
students_performance_filtered= students_performance_filtered.replace({"yes": 1, "no": 0})
students_performance_filtered['sex']= students_performance_filtered['sex'].replace({"F": 1, "M": 0})


# pearson coef
corr_uncleaned, p_value_uncleaned=stats.pearsonr(students_performance_unclean["G3"], students_performance_unclean["absences"])
corr_filtered, p_value_filtered=stats.pearsonr(students_performance_filtered["G3"], students_performance_filtered["absences"])
print(f'''
    Correlation and P value for unfiltered:{corr_uncleaned, p_value_uncleaned}
    Correlation and P value for filtered:{corr_filtered, p_value_filtered}
      ''')
#it shows that there is  meaningful connection between students who had high absences and score on the G3, which is evident by the p value


#task 3



numeric_col=students_performance_filtered.select_dtypes(include='number').columns.tolist()
numeric_col_corr={}

for col in numeric_col:
    corr, p_value=stats.pearsonr(students_performance_filtered["G3"], students_performance_filtered[col])
    numeric_col_corr[col]=[float(corr)]
# print(numeric_col_corr)
 
sorted_by=dict(sorted(numeric_col_corr.items(), key=lambda x: x[1]))

print(sorted_by)
#what I find the most surprising is there is not a strong correlations between studytime and travel time and there seems to be a stronger correlation between FEDU than MEDU
students_performance_filtered=students_performance_filtered.sort_values(by="G3")
plt.bar(students_performance_filtered['G1'], students_performance_filtered['G3'])
plt.title("G1 vs G3")
plt.xlabel('G3')
plt.ylabel('G1')
plt.tight_layout()
# plt.show()
plt.savefig('assignments_02/outputs/G1_vs_G3.png')

#based on these graph you can visualize the very strong correlation between previous semester grades and the file grade, so it really doe matter how the tear starts 

plt.bar(students_performance_filtered['G2'], students_performance_filtered['G3'])
plt.title("G2 vs G3")
plt.xlabel('G3')
plt.ylabel('G2')
plt.tight_layout()
plt.show()
plt.savefig('assignments_02/outputs/G2_vs_G3.png')

#There is a stronger ocrrelations between g3 and g2 that in G1 evidant by the difference in the magnitude of the bars. based on these graph you can visualize the very strong correlation between previous semester grades and the file grade 


# task 4
failures= np.array(students_performance_filtered["failures"]).reshape(-1, 1)
G3= np.array(students_performance_filtered["G3"]).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    failures, G3, test_size=0.2, random_state=42
)

model = LinearRegression()                   
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)

print(f'''
    Slope: {model.coef_[0][0]}
    Intercept: {model.intercept_}
    RMSE: {np.sqrt(np.mean((y_pred - y_test) ** 2))}
    R^2: {model.score(X_test, y_test)}
''')

#the slope with worse than I expected,it means the number of failures a student has the inverse relationship on grade so more failure mean
#lower grade, I expected to be bad but this is less than -1
#R2 is lower than I expect sicne the nnumber is so low

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime", 'G1']
X = students_performance_filtered[feature_cols].values
y = students_performance_filtered["G3"].values

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model_full = LinearRegression()                   
model_full.fit(X_train_d, y_train_d)  
y_pred = model_full.predict(X_test_d)

print(f'''
    g1
    Slope: {model_full.coef_[0]}
    Intercept: {model_full.intercept_}
    RMSE: {np.sqrt(np.mean((y_pred - y_test_d) ** 2))}
    R^2: {model_full.score(X_test_d, y_test_d)}
''')
#adding more features helps but there is still a strong relationship
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef}")

#schoolsup is the most surprising results siince it is has the highest correlations magnititube 
    #depite it being an inverse relationship but I think that is because student who were not performing they probably needed more school assistance compare to an sverage student
# there is a pretty significant grab between train and test
# I would likely drop the coef with lower slopes like FEDU, MEDU and actvities then check how well the R2 is from there and iteriate
    

#task 6

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test_d, alpha=0.6, color="steelblue", edgecolors="white", linewidths=0.5)

#prediction line
plt.plot([y_test_d.min(), y_test_d.max()], [y_test_d.min(), y_test_d.max()], label="Prediction line")

plt.xlabel("Predicted (y_hat)")
plt.ylabel("Actual (y)")
plt.title("Predicted vs Actual - Full Model")
plt.legend()
plt.tight_layout()
plt.savefig('assignments_02/outputs/predicted_vs_actual.png')
plt.show()

# Which two features have the largest positive and largest negative coefficients, and what those mean
# One result that surprised you

# a model error that large would be there needs to be significant look at which variable help support the model and which are deterrents
#larget + is Interet an study time which makes sense, more time that is spent stuying and access to information increase your grable
#largest - is failures, which tracks with the overall grades and Schoolup so additional schooling meant students needed help and therefore performed worse

#sheesh, that made the whole model look 10x more accurate, evident through my previous visulaization G1 an G3 has such a strong correlation
#G1 is not causing G3 but an early predictor and should be used to identift at risk students earlier
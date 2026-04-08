import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from prefect import task, flow
from prefect.logging import get_run_logger 
#Pandas Warmup Exercises

#Pandas Question 1

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
print(df)
print(f'''
    first three rows: 
    {df.head(3)}
    shape: 
    {df.shape}
    Datatype of each col:
    {df.info()}
      ''')

#Pandas Question 2
passed_and_above_80 = df[(df['grade']>80) & (df['passed']==True)]
print(passed_and_above_80)

#Pandas Question 3
df['curved']=df['grade']+5
print(df)

#Pandas Question 4
df['name_upper']=df['name'].apply(str.upper)
print(df[['name', 'name_upper']])

#Pandas Question 5
city_average= df.groupby('city')['grade'].mean()
print(city_average)

#Pandas Question 6
df['city']= df['city'].replace('Austin','Houston')
print(df[['name', 'city']])

#Pandas Question 7
sorted_df= df.sort_values(ascending=False, by='grade')
print(sorted_df.head(3))

#NumPy Review
#NumPy Question 1

one_num=np.array([10, 20, 30, 40, 50])
print(f'''
    ndim: 
    {one_num.ndim}
    shape: 
    {one_num.shape}
    dtype:
    {one_num.dtype}
      ''')

#NumPy Question 2
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])


print(f'''
    size: 
    {arr.size}
    shape: 
    {arr.shape}
      ''')

#NumPy Question 3
print(arr[0:2, 0:2])

#NumPy Question 4
zero_np= np.zeros((3,4))
ones_np= np.ones((2,5))
print(zero_np)
print(ones_np)

#NumPy Question 5
num_arr=np.arange(0, 50, 5)
print(f'''
    shape:{num_arr.shape}
    mean:{np.mean(num_arr)}
    sum:{np.sum(num_arr)}
    Std:{np.std(num_arr)}
    ''')

#NumPy Question 6
random_arr=np.random.normal(size=200)
print(random_arr)
print(f'''
    mean:{np.mean(random_arr)}
    Std:{np.std(random_arr)}
    ''')

#Matplotlib Question 1

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]


plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#Matplotlib Question 2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects, scores, color="blue")
plt.title("Grades")
plt.xlabel("subjects")
plt.ylabel("scores")
plt.show()

# Matplotlib Question 3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, label="Set 1", color="blue")
plt.scatter(x2, y2, label="Set 2", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title('Set 1 and Set 2')
plt.show()

#Matplotlib Question 4
fig, (ax1, ax2)=plt.subplots(1, 2)
ax1.plot(x1, y1)
ax1.set_title("Set 1")
ax2.plot(x2, y2)
ax2.set_title("Set 2")
plt.tight_layout()
plt.show()

#Descriptive Statistics Review
#Descriptive Stats Question 1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f'''
     Mean:{np.mean(data)}
     Median:{np.median(data)}
     Std:{np.std(data)}
     Variance:{np.var(data)}
     ''')

#Descriptive Stats Question 2

random_500= np.random.normal(65, 10, 500)
plt.hist(random_500, bins=20, color="red", edgecolor='black')
plt.title("Histogram")
plt.xlabel("Random Numbers")
plt.ylabel("Frequency")
plt.show()

#Descriptive Stats Question 3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

fig, (ax1, ax2)=plt.subplots(2)
ax1.boxplot(group_a, orientation="horizontal", tick_labels=["Group A"])
ax2.boxplot(group_b,  orientation="horizontal", tick_labels=["Group B"])
fig.suptitle('Score Comparison')
plt.tight_layout()
plt.show()

#Descriptive Stats Question 4

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
fig, (ax1, ax2)=plt.subplots(1,2)
ax1.boxplot(normal_data, orientation="horizontal", tick_labels=["Normal"])
ax2.boxplot(skewed_data,  orientation="horizontal", tick_labels=["Exponential"])
fig.suptitle('Distribution Comparison')
plt.tight_layout()
plt.show()
#The exponential graph is more skewed since the left whisper is shorter and the median line is closer to zero. If the exponential box was more symmetrical then the central tendency would be towards the mean

#Descriptive Stats Question 5

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
mode1=stats.mode(np.array(data1))
mode2=stats.mode(np.array(data2))
print(f'''
 Data1:
    Mean:{np.mean(data1)}
    Median:{np.median(data1)}
    Mode:{mode1.mode}
 Data2:
    Mean:{np.mean(data2)}
    Median:{np.median(data2)}
    Mode:{mode2.mode} 
''')
#The mean is skewed larger in dataset 2 because of the value 150, which it is an outlier
#The medians are the same because simply the middle of a dataset and since they have the same number of elements it is easy for the middle number to be the same

#Hypothesis Question 1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print('#Hypothesis Question 1')
print("t-statistic:", t_stat)
print("p-value:", p_val)

#Hypothesis Question 2
alpha = 0.05
print('#Hypothesis Question 2')
if p_val < alpha:
    print("The difference is statistically significant")
else:
    print(" No statistically significant difference detected.")

#Hypothesis Question 3
    
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_val = stats.ttest_rel(before, after)
print('#Hypothesis Question 3')
print(f't-statistic:{t_stat}, p-val:{p_val}')

#Hypothesis Question 4
print('#Hypothesis Question 4')
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_val = stats.ttest_1samp(scores, popmean=70)
print(f't-statistic:{t_stat}, p-val:{p_val}')

#Hypothesis Question 5
t_stat, p_val = stats.ttest_ind(group_a, group_b, alternative='less')
print("#Hypothesis Question 5")
print(f'p-val:{p_val}')

# Hypothesis Question 6
print('Since the be value is extremely small and alternative was set to less that means that group A was actually less than group be cause because a smaller p value signifies less correlation')

#Correlation Review
#Correlation Question 1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_martix=np.corrcoef(x,y)
print(f'Correlation Matrix: {corr_martix}')
print(f'Correlation Coefficient: {corr_martix[0]}')
print("The correlations is the same be positive and linear since they are both consistently increasing and a steady rate")

#Correlation Question 2

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr, p_value=stats.pearsonr(x,y)
print(f"Correlation:{corr}, p_value: {p_value}")

#Correlation Question 3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(f'{df.corr()}')


#Correlation Question 4

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y, color='teal')
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Correlation Question 5

sns.heatmap(df, annot=True, cmap="Purples" )
plt.title("Correlation Heatmap")
plt.show()

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


#Pipeline Question 1

def create_series(arr)->pd.Series:
    values = pd.Series(arr)
    return values


def clean_data(series)->pd.Series:
    cleaned_series=series.dropna()
    return cleaned_series


def summarize_data(series)-> dict:
    mode= float(series.mode()[0])
    stats={"mean": float(series.mean()),
           "median":float(series.median()),
           "std":float(series.std()),
           "mode":mode
           }
    return stats

def data_pipeline(arr):
    series=create_series(arr)
    cleaned_series=clean_data(series)
    results=summarize_data(cleaned_series)
    print(results)
    return results

if __name__ == "__main__":
    data_pipeline(arr)
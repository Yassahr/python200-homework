import pandas as pd
import numpy as np
from prefect import task, flow


#Pipeline

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

#Pipeline Question 2
@task()
def create_series(arr)->pd.Series:
    values = pd.Series(arr)
    return values

@task()
def clean_data(series)->pd.Series:
    cleaned_series=series.dropna()
    return cleaned_series

@task()
def summarize_data(series)-> dict:
    mode= float(series.mode()[0])
    stats={"mean": float(series.mean()),
           "median":float(series.median()),
           "std":float(series.std()),
           "mode":mode
           }
    return stats

@flow()
def pipeline_flow(arr):
    series=create_series(arr)
    cleaned_series=clean_data(series)
    results=summarize_data(cleaned_series)
    print(results)
    return results

if __name__ == "__main__":
    pipeline_flow(arr)

# Question 1 
    # The reason why the pipeline is worth is because if you have ETL processes that run nightly it allows you to encapsulate these into 
    #functions that allows it to be scaled and reduces redundacy. A standard input can be put in and an reliable output is returned

#Question 2
    #An example of when this may be useful is for a hospitals system. If there is a large amount of data they need to integrate into their system on a regular basis to ensure
    #patient information is up to date and standardized they can run a similar pipeline to clean the data and integrate into different databases
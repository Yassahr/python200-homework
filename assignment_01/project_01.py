import pandas as pd
import numpy as np
from prefect import task, flow
from prefect.logging import get_run_logger
from prefect.runtime import task_run
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

#helper function
def csv_to_df(x)->pd.DataFrame:
    filename=f"../python-200/assignments/resources/happiness_project/world_happiness_{x}.csv"
    dataframe= pd.read_csv(filename, sep=";", decimal=',')
    dataframe.head(5)
    dataframe['Year']=x
    return dataframe

@task
def summary():
    logger = get_run_logger()
    logger.info('''
    This data set spanned from 174 countries and 9 years. 
    The factor that has the highest bearing on happiness was social support, made evident by the heatmap and high correlation.
    Countries with the Highest Happiness ranking were: Switzerland, Morocco, and South Africa   
    Countries with the Highest Happiness ranking were: Congo, Eswatini, and Turkiye 
    One average there was no a statistical difference in the world when it came to 2019 v 2020 happiness but I belive further aggregations would need to be done to get the full picture.    
    ''')

@task
def correlation(df)->None:
    logger = get_run_logger()

    happiness=df['Happiness score']
    gdp=df['GDP per capita']
    ss=df['Social support']
    life=df['Healthy life expectancy']
    choice=df['Freedom to make life choices']
    gen=df['Generosity']
    poc=df['Perceptions of corruption']

    gdp_corr, p = stats.pearsonr(gdp, happiness)
    ss_corr, p = stats.pearsonr(ss, happiness)
    life_corr, p = stats.pearsonr(life, happiness)
    choice_corr, p = stats.pearsonr(choice, happiness)
    gen_corr, p = stats.pearsonr(gen, happiness)
    poc_corr, p = stats.pearsonr(poc, happiness)

    logger.info(f'''
        gdp_corr: {gdp_corr} p_val:{p}  
        ss_corr: {ss_corr} p_val:{p}       
       life_corr: {life_corr} p_val:{p}  
       choice_corr: {choice_corr} p_val:{p}  
       gen_corr: {gen_corr} p_val:{p}  
       poc_corr: {poc_corr} p_val:{p} 
       adjusted_alpha = {0.05/6}
        No values are below the adjusted alpha so none are significant. If the alpha was not adjusted all of the p vlaues would be considered significant
            ''')


@task
def hypothesis(df)->None:
    logger = get_run_logger()
    happiness_2019= df[df['Year']==2019]['Happiness score']
    happiness_2020= df[df['Year']==2020]['Happiness score']
    t_stat, p_val = stats.ttest_ind(happiness_2019, happiness_2020)
    if p_val < 0.05:
        logger.info("The difference is statistically significant difference between the happiness is 2020 and 2019")
    else:
        logger.info(" No statistically significant difference detected.")
    happiness_ssa= df[df['Regional indicator']=='Sub-Saharan Africa']['Happiness score']
    happiness_mena= df[df['Regional indicator']=='Middle East and North Africa']['Happiness score']
    t_stat, p_val = stats.ttest_ind(happiness_ssa, happiness_mena)
    if p_val < 0.05:
        logger.info("The difference is statistically significant difference between the happiness is Sub Sahran Africa and North Africa/Middle East")
    else:
        logger.info(" No statistically significant difference detected.")


@task
def visualizations(df):
    logger = get_run_logger()
    #histogram
    plt.hist(df['Happiness score'], bins=20, color="purple", edgecolor="black")
    plt.title('Happiness Historgram')
    plt.savefig('assignment_01/outputs/happiness_histogram.png')
    plt.show()
    logger.info("happiness_histogram.png saved")

    #boxplots
    df.boxplot(column='Happiness score', by='Year')
    plt.savefig('assignment_01/outputs/happiness_by_year.png')
    plt.show()
    logger.info("happiness_by_year.png saved")


    #scatterplot
    plt.scatter(df['Year'], df['Happiness score'], label="Happiness", color="blue", alpha=0.4)
    plt.scatter(df['Year'], df['GDP per capita'], label="GDP", color="red", alpha=0.4)
    plt.xlabel("Year")
    plt.title('GDP vs Happiness')
    plt.tight_layout()
    plt.savefig('assignment_01/outputs/gdp_vs_happiness.png')
    plt.show()
    logger.info("gdp_vs_happiness.png saved")


    #heatmap
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('assignment_01/outputs/correlation_heatmap.png')
    plt.show()
    logger.info("correlation_heatmap.png saved")


@task
def get_stats(clean_df):
    logger = get_run_logger()

    overall_happiness_mean=clean_df['Happiness score'].mean()
    overall_happiness_std=clean_df['Happiness score'].std()
    overall_happiness_median=clean_df['Happiness score'].median()
    happiness_mean_by_year_region=clean_df.groupby(by=['Year','Regional indicator'])['Happiness score'].mean()


    logger.info(f'''
                Happiness Mean:{overall_happiness_mean}
                Happiness STD:{overall_happiness_std}
                Happiness Median:{overall_happiness_median}
                Agg happiness:{happiness_mean_by_year_region}
                ''')
@task
def clean_data(dataframe)->pd.DataFrame:
    dataframe=dataframe.replace("NaN", None)
    dataframe=dataframe.dropna()
    return dataframe

@task(retries=3, retry_delay_seconds=2)
def cumulate_files()->pd.DataFrame:
    logger = get_run_logger()
    happiness_df_uncleaned=[]
    for i in range(2015, 2024):
        logger.info(f"Read CSV for {i} year")
        df=csv_to_df(i)
        happiness_df_uncleaned.append(df)
    all_year=pd.concat(happiness_df_uncleaned, ignore_index=True)
    all_year.to_csv("assignment_01/outputs/merged_happiness.csv", mode="w", index=False)
    return all_year


@flow
def happiness_pipeline():
    df= cumulate_files()
    df= clean_data(df)
    get_stats(df)
    visualizations(df)
    hypothesis(df)
    correlation(df)
    summary()


if __name__ == "__main__":
    happiness_pipeline()


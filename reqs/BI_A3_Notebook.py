#!/usr/bin/env python
# coding: utf-8

# # Assignment 3: Data Analytics
# 188.429 Business Intelligence WS2021
# 
# by Gunnar Sjúrðarson Knudsen & Katrin Schreiberhuber
# 
# For our final project of the course, we have decided to use the dataset called **Productivity Prediction of Garment Employees Data Set**. This dataset contains records of the productivity of a team of workers on different days. The goal of this project is therefore to model the actual productivity of the workers and to get an idea of the influencing factors.
# 
# **source of the dataset:**
# 
# https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees#
# 

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Setup" data-toc-modified-id="Setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Setup</a></span><ul class="toc-item"><li><span><a href="#Required-libraries" data-toc-modified-id="Required-libraries-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Required libraries</a></span></li><li><span><a href="#Visualization-settings" data-toc-modified-id="Visualization-settings-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Visualization settings</a></span></li><li><span><a href="#Constants" data-toc-modified-id="Constants-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Constants</a></span></li><li><span><a href="#Read-in-data" data-toc-modified-id="Read-in-data-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Read in data</a></span></li></ul></li><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span><ul class="toc-item"><li><span><a href="#Column-Description" data-toc-modified-id="Column-Description-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Column Description</a></span></li><li><span><a href="#First-glance" data-toc-modified-id="First-glance-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>First glance</a></span></li><li><span><a href="#Categorial-vs-Numeric-Attributes" data-toc-modified-id="Categorial-vs-Numeric-Attributes-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Categorial vs Numeric Attributes</a></span></li><li><span><a href="#Nullity-of-columns" data-toc-modified-id="Nullity-of-columns-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Nullity of columns</a></span></li><li><span><a href="#Correlation-between-target-and-actual" data-toc-modified-id="Correlation-between-target-and-actual-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Correlation between target and actual</a></span></li><li><span><a href="#Pairwise-Correlations" data-toc-modified-id="Pairwise-Correlations-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Pairwise Correlations</a></span></li><li><span><a href="#Understand-the-difference-between-Actual-and-target" data-toc-modified-id="Understand-the-difference-between-Actual-and-target-2.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>Understand the difference between Actual and target</a></span></li></ul></li><li><span><a href="#Preprocessing" data-toc-modified-id="Preprocessing-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Data-Cleansing" data-toc-modified-id="Data-Cleansing-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Data Cleansing</a></span></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Feature Engineering</a></span></li><li><span><a href="#Train/Test/Val-Split" data-toc-modified-id="Train/Test/Val-Split-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Train/Test/Val Split</a></span></li></ul></li><li><span><a href="#Build-models" data-toc-modified-id="Build-models-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Build models</a></span><ul class="toc-item"><li><span><a href="#Setup-of-evaluation" data-toc-modified-id="Setup-of-evaluation-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Setup of evaluation</a></span></li><li><span><a href="#Linear-Regression" data-toc-modified-id="Linear-Regression-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Linear Regression</a></span></li><li><span><a href="#Lasso-Regression" data-toc-modified-id="Lasso-Regression-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Lasso Regression</a></span></li><li><span><a href="#Ridge-Regression" data-toc-modified-id="Ridge-Regression-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Ridge Regression</a></span></li><li><span><a href="#Random-Forrest" data-toc-modified-id="Random-Forrest-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Random Forrest</a></span></li><li><span><a href="#Support-Vector-Regression" data-toc-modified-id="Support-Vector-Regression-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Support Vector Regression</a></span></li><li><span><a href="#K-Nearest-Neighbours" data-toc-modified-id="K-Nearest-Neighbours-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>K Nearest Neighbours</a></span></li><li><span><a href="#Extreme-Gradient-Descent-Boosting" data-toc-modified-id="Extreme-Gradient-Descent-Boosting-4.8"><span class="toc-item-num">4.8&nbsp;&nbsp;</span>Extreme Gradient Descent Boosting</a></span></li><li><span><a href="#Gradient-Boosting" data-toc-modified-id="Gradient-Boosting-4.9"><span class="toc-item-num">4.9&nbsp;&nbsp;</span>Gradient Boosting</a></span></li></ul></li><li><span><a href="#Evaluate-models" data-toc-modified-id="Evaluate-models-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Evaluate models</a></span><ul class="toc-item"><li><span><a href="#Hyperparameter-search" data-toc-modified-id="Hyperparameter-search-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Hyperparameter search</a></span></li></ul></li><li><span><a href="#Retrain-the-model-with-identical-hyperparameters-using-the-full-train-and-test-set" data-toc-modified-id="Retrain-the-model-with-identical-hyperparameters-using-the-full-train-and-test-set-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Retrain the model with identical hyperparameters using the full train and test set</a></span><ul class="toc-item"><li><span><a href="#Concatenate-Train-and-Test" data-toc-modified-id="Concatenate-Train-and-Test-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Concatenate Train and Test</a></span></li><li><span><a href="#Rebuild-model-on-full-dataset" data-toc-modified-id="Rebuild-model-on-full-dataset-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Rebuild model on full dataset</a></span></li><li><span><a href="#compare-performance-on-different-training" data-toc-modified-id="compare-performance-on-different-training-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>compare performance on different training</a></span></li></ul></li><li><span><a href="#Explainability" data-toc-modified-id="Explainability-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Explainability</a></span><ul class="toc-item"><li><span><a href="#All-features" data-toc-modified-id="All-features-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>All features</a></span></li><li><span><a href="#Only-the-most-important-ones" data-toc-modified-id="Only-the-most-important-ones-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Only the most important ones</a></span></li></ul></li><li><span><a href="#Run-on-test" data-toc-modified-id="Run-on-test-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Run on test</a></span></li><li><span><a href="#Run-on-validation" data-toc-modified-id="Run-on-validation-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Run on validation</a></span></li><li><span><a href="#Analysis-of-performance-in-detail" data-toc-modified-id="Analysis-of-performance-in-detail-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Analysis of performance in detail</a></span></li></ul></div>

# ## Setup

# ### Required libraries

# In[1]:


# Standard Data Science toolkit
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

# Visualization toolkits
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# Machine Learning / Modelling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

import random
random.seed(420)
np.random.seed(69)


# ### Visualization settings

# In[2]:


sns.set_style(style='white')
sns.set(rc={
    'figure.figsize': (12,7),
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '.9',
    'axes.linewidth': 1.0,
    'grid.linestyle': u'-'},
    font_scale=1.5)
custom_colors=["#3498db", "#95a5a6","#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)
background_color='#fbfbfb'


# ### Constants

# In[3]:


DATA_LOCATION = 'raw_data/garments_worker_productivity.csv'
GRAPHICS_LOCATION = 'output_graphics/'


# ### Read in data

# In[4]:


df = pd.read_csv(DATA_LOCATION, parse_dates=['date'])#, index_col=['date'])
df.set_index('date', drop = False, inplace=True)
df


# ## Exploratory Data Analysis
# Get a feel for what the data contains, so that we know which preprocessing is needed

# ### Column Description
# _Taken directly from the source_
#   * 01 `date` : Date in MM-DD-YYYY
#   * 02 `day` : Day of the Week
#   * 03 `quarter` : A portion of the month. A month was divided into four quarters
#   * 04 `department` : Associated department with the instance
#   * 05 `team_no` : Associated team number with the instance
#   * 06 `no_of_workers` : Number of workers in each team
#   * 07 `no_of_style_change` : Number of changes in the style of a particular product
#   * 08 `targeted_productivity` : Targeted productivity set by the Authority for each team for each day.
#   * 09 `smv` : Standard Minute Value, it is the allocated time for a task
#   * 10 `wip` : Work in progress. Includes the number of unfinished items for products
#   * 11 `over_time` : Represents the amount of overtime by each team in minutes
#   * 12 `incentive` : Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action.
#   * 13 `idle_time` : The amount of time when the production was interrupted due to several reasons
#   * 14 `idle_men` : The number of workers who were idle due to production interruption
#   * 15 `actual_productivity` : The actual % of productivity that was delivered by the workers. It ranges from 0-1. 

# ### First glance
# First step is to just apply standard pandas functions to see what we are working with:

# In[5]:


print(f'Shape: {df.shape}')
# print("------------------------------------------------")
display(df.describe())
# print("------------------------------------------------")
# print("datatypes:")
# print(df.dtypes)
# print("------------------------------------------------")
# print("Skew:")
# print(df.skew())
# print("------------------------------------------------")
# display(df.info())


# **Intermediary Conclusion:** Blaaaerb.... Nothing too exciting to see here

# ### Categorial vs Numeric Attributes
# Splitting these up, for separate exploration

# In[6]:


category = df.select_dtypes(include='object')
categorial_columns = category.columns
numerical = df.select_dtypes(exclude='object')
numerical_columns = numerical.columns

print("Categorical Attributes:")
for col in category.columns:
    print(f"{col}")
    print(category[col].unique())
    print()

print("Numerical Attributes:")
for n in numerical.columns:
    print(n)


# **Insights:** Team is not supposed to be a numerical attribute

# In[7]:


#pd.merge(
    
all_dates = pd.date_range(start=df.date.min(), end=df.date.max(), freq=None).to_series(name = "date")
dates_count = df.groupby(df.index.date).count()["department"]
dfx = pd.concat([all_dates, dates_count], axis=1)
dfx.columns = ["date", "no of data entries"]
dfx.date = dfx.date.dt.date
#dfx.plot.bar(x = "date", y = "no of data entries");
plt.bar(dfx.date, dfx["no of data entries"])
plt.title("Data entries per day")
plt.savefig(GRAPHICS_LOCATION + 'data_entries_per_day.png')
plt.show()


# In[8]:


df["date"] = pd.to_datetime(df['date'], infer_datetime_format= True)
df["dayOfMonth"] =  df.date.dt.day


# In[9]:


# perform groupby
sns.displot(df, x="dayOfMonth", hue="quarter")
plt.savefig(GRAPHICS_LOCATION + 'quarters_per_month.png')
plt.show()


# In[10]:


df.groupby("quarter").mean()["actual_productivity"].plot.bar(title = "Productivity of workers across the month")


# In[11]:


f, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=df, x='team', y='no_of_style_change', ax=ax)#, hue='department', ax=ax)
plt.savefig(GRAPHICS_LOCATION + 'EDA_styleChangesPerTeam.png')
plt.show()


# #### Categorical Distribution

# In[12]:


for i in range(len(categorial_columns)):
    df[categorial_columns[i]].value_counts().plot(kind='bar', title=categorial_columns[i])
    plt.savefig(GRAPHICS_LOCATION + 'EDA_Categorical_Distribution_' + categorial_columns[i] + '.png')
    plt.show()


# **Intermediary Conclusion:** 
# - Q5 of the month is always shorter than the others as it is what is left after 4 weeks
# - There is no work on a friday
# - mistakes in labelling of sewing and finishing
# - missing entry for 21.2.2015 for whatever reason

# #### Numerical Distribution

# In[13]:


for i in range(len(numerical.columns)):
    if numerical.columns[i] == 'date':
        print('Skipping')
    else:
        numerical.iloc[:, i].plot(kind='hist', title=numerical.columns[i])
        plt.savefig(GRAPHICS_LOCATION + 'EDA_Numerical_Distribution_' + numerical.columns[i] + '.png')
        plt.show()


# In[14]:


numerical.boxplot(column = [ 'targeted_productivity', 'actual_productivity'],figsize=(8,7))
plt.savefig(GRAPHICS_LOCATION + 'EDA_Numerical_boxplot_1.png')
plt.show()

numerical.boxplot(column=['wip', 'over_time', 'incentive'],figsize=(8,7))
plt.savefig(GRAPHICS_LOCATION + 'EDA_Numerical_boxplot_2.png')
plt.show()

numerical.boxplot(column = ['smv'],figsize=(8,7))
plt.savefig(GRAPHICS_LOCATION + 'EDA_Numerical_boxplot_3.png')
plt.show()

numerical.boxplot(column = [ 'idle_time', 'idle_men',  'no_of_workers'] ,figsize=(8,7))
plt.savefig(GRAPHICS_LOCATION + 'EDA_Numerical_boxplot_4.png')
plt.show()


# **Intermediary Conclusion:** 
# 
# - targeted productivity is max 0.8
# - actual productivity has a very wide range
# - most variables very narrow except for small outliers

# #### Categorical In disguise
# We see that team number should be treated as a categorical variable

# In[15]:


# Team feature is a categorical variable
team_count = df.team.value_counts()
plt.figure(figsize=(10,5))
sns.barplot(x = team_count.index, y = team_count.values, color = "orange")
plt.title('Team Numbers')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Teams', fontsize=12)
plt.savefig(GRAPHICS_LOCATION + 'EDA_Categorical_In_disguise_Teams.png')
plt.show()


# **Intermediary Conclusion:** Bllaaaaerb

# ### Nullity of columns
# Let's start by seeing how our nulls are distributed

# In[16]:


#msno.bar(df)
#plt.show()

msno.matrix(df)
plt.savefig(GRAPHICS_LOCATION + 'EDA_null_matrix.png')
plt.show()

#msno.heatmap(df)
#plt.show()


# We see that only one column has missing values; let's see if we can figure out a pattern for this

# In[17]:


fig = px.line(df, x=df.index, y="wip", color = "team")
fig.update_layout(yaxis_range=[0,2000])
fig.show()


# **Intermediary Conclusion:** It seems to only be the column `wip` _(Work in progress. Includes the number of unfinished items for products)_ that contains nulls, and they seem to be **Missing At Random**
# Reason being that when looking at productivity per team, there seems to be pattern in. Therefore we can NOT fill it with zeros, but should instead do groupwise time interpolation

# ### Correlation between target and actual
# Let's see how the targeted productivity corresponds to the actual productivity

# #### Distribution differences

# In[18]:


plt.figure(figsize = (16,6))
ax=sns.lineplot(y='targeted_productivity',x='date' ,data = df, legend='brief', label = 'target')
ax=sns.lineplot(y= 'actual_productivity',x='date',data=df, legend = 'brief', label = "actual")
ax.set(ylabel = 'Productivity')
ax.legend()
plt.savefig(GRAPHICS_LOCATION + 'EDA_target_vs_actual_TimeSeries.png')
plt.show()

sns.histplot(data=df[['targeted_productivity', 'actual_productivity']], element='poly')
plt.savefig(GRAPHICS_LOCATION + 'EDA_target_vs_actual.png')
plt.show()
# **Intermediary Conclusion:** Don't know yet... Most of the time, target seems higher than actual
# 
# There is definitely a correlation

# #### Correlations to the actual productivity
# Let's see what affects the actual productivity

# In[19]:


corrMatrix = df.corr()
plt.figure(figsize=(5, 10))
heatmap = sns.heatmap(corrMatrix[['actual_productivity']].sort_values(by='actual_productivity', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with actual_productivity', fontdict={'fontsize':18}, pad=16);
plt.savefig(GRAPHICS_LOCATION + 'EDA_Attribute_Correlation_With_Actual.png')
plt.show()


# **Intermediary Conclusion:** Target seems to be a decent indicator, meaning that workers do speed up when target is high. we also see that the number of style change request negatively effects it...

# ### Pairwise Correlations

# #### Looking at them visually

# In[20]:


sns.pairplot(df)
plt.savefig(GRAPHICS_LOCATION + 'EDA_Pairwise_Correlation_Matrix.png')
plt.show()


# **Intermediary Conclusion:** Oookay These are too many columns to visualize. However, it is easy to see that e.g. Quarter, date, and day all represent partially the same, and could be excluded. Probably other data could as well... Lets get back to that later

# #### List of important correlations
# As the above was a bit unreadable, let's try to create a list of most correlated attributes

# ##### Pearson
# _from: https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/regression/supporting-topics/basics/a-comparison-of-the-pearson-and-spearman-correlation-methods/_
# The Pearson correlation evaluates the linear relationship between two continuous variables. A relationship is linear when a change in one variable is associated with a proportional change  in the other variable.
# 
# For example, you might use a Pearson correlation to evaluate whether increases in temperature at your production facility are associated with decreasing thickness of your chocolate coating.
# 
# ##### Kendall
# _from https://www.statsdirect.com/help/nonparametric_methods/kendall_correlation.htm_
# Kendall's rank correlation provides a distribution free test of independence and a measure of the strength of dependence between two variables.
# 
# Spearman's rank correlation is satisfactory for testing a null hypothesis of independence between two variables but it is difficult to interpret when the null hypothesis is rejected.  Kendall's rank correlation improves upon this by reflecting the strength of the dependence between the variables being compared.
# 
# ##### Spearman
# _from: https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/regression/supporting-topics/basics/a-comparison-of-the-pearson-and-spearman-correlation-methods/_
# The Spearman correlation evaluates the monotonic relationship between two continuous or ordinal variables. In a monotonic relationship, the variables tend to change together, but not necessarily at a constant rate. The Spearman correlation coefficient is based on the ranked values for each variable rather than the raw data.
# 
# Spearman correlation is often used to evaluate relationships involving ordinal variables. For example, you might use a Spearman correlation to evaluate whether the order in which employees complete a test exercise is related to the number of months they have been employed.

# In[21]:


# Create functions for calculating
## Inspired/stolen from https://stackoverflow.com/questions/54207492/creating-a-list-from-a-correlation-matrix-in-pandas
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, correlation_type = 'spearman', abs_only = False, n=500):
    df_corr = df.corr(method=correlation_type)
    #display(df_corr.style.background_gradient(cmap="Blues"))
    au_corr = df_corr
    if abs_only:
        au_corr = au_corr.abs()
    au_corr = au_corr.unstack()
    labels_to_drop = get_redundant_pairs(df_corr)
    #au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    #print("Top Absolute Correlations")
    #print(au_corr[0:n])
    return au_corr[0:n]

def compare_correlations(df):
    corr_spearman = get_top_abs_correlations(df, correlation_type = 'spearman', abs_only = False, n=300)
    corr_pearson = get_top_abs_correlations(df, correlation_type = 'pearson', abs_only = False, n=300)
    corr_kendall = get_top_abs_correlations(df, correlation_type = 'kendall', abs_only = False, n=300)
    
    corr_spearman = pd.DataFrame(corr_spearman, columns=['spearman'])
    corr_pearson = pd.DataFrame(corr_pearson, columns=['pearson'])
    corr_kendall = pd.DataFrame(corr_kendall, columns=['kendall'])
    
    returned_df = corr_spearman.join(corr_pearson)
    returned_df = returned_df.join(corr_kendall)
    
    #display(returned_df.style.background_gradient(cmap="Blues"))
    
    return returned_df

rdf = compare_correlations(df)

rdf = rdf[np.in1d(rdf.index.get_level_values(1), ['actual_productivity'])]

display(rdf.reindex(rdf.mean(axis=1).abs().sort_values(ascending=False).index, axis=0).droplevel(1, axis=0).style.background_gradient(cmap="Blues"))


# **Intermediary Conclusion:** Irregardless of which method we choose, there seems to be quite a good structure to which columns are correlated.

# ### Understand the difference between Actual and target
# One way to visualize this, is to calculate the difference between these values, and visually inspect when it changes

# In[22]:


# Add margin
df['margin'] = df['actual_productivity'] - df['targeted_productivity']

# Show plot
f, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=df, x='team', y='margin', hue='department', ax=ax)
plt.savefig(GRAPHICS_LOCATION + 'EDA_Difference_in_margin.png')
plt.show()

# Delete again
df.drop(columns = 'margin',inplace = True)


# **Intermediary Conclusion:** There are definitely some teams that tend to overshoot the target compared to some who usualy fail to meet the target

# ## Preprocessing
# Next step is preparing for modelling

# ### Data Cleansing
# Fixing the various faults found earlier

# #### Department
# This column contains a erroneous space, as well as a typo

# In[23]:


# Original values:
print(df.department.unique())

# Cleanse
## Remove space
df['department'] = df.department.str.strip()
## Replace typo
df['department'] = df.department.replace(['sweing'],['sewing'])

# Check values again:
print(df.department.unique())


# #### Fix `WIP` missing values
# We think WIP is missing at random. Therefore we interpolate over time within each team

# In[24]:


#df['wip'].interpolate(method='time', inplace=True)
#df["wip_zeros"] = df["wip"].fillna(0)
df['wip'] = df.groupby('team')['wip'].apply(lambda g: g.interpolate(method='time'))
# Fix starting positions
df['wip'] = df['wip'].fillna(method='bfill')


# ### Feature Engineering
# Next up is creating some missing features, and or re-encoding

# #### Add Month Column

# In[25]:


df['month'] = df['date'].dt.month_name() 
# add day of the month a column ( already done above, but mentioned here for completeness)
#df["dayOfMonth"] =  df.date.dt.day


# #### Modify Incentive
# Add a tiny amount for better modelling

# In[26]:


df.loc[df.incentive==0, 'incentive'] = 0.0001


# #### Man power per smv

# In[27]:


df['smv_manpower'] = np.log(df['smv'] / df['no_of_workers'])


# #### One-hot enconding
# Change to strings, to we get use pandas `get_dummies`

# In[28]:


# Change to string
df['team']=df['team'].astype(str)
# df['no_of_style_change']=df['no_of_style_change'].astype(str)
# One-hot-encode
df=pd.get_dummies(df)
# Drop date
df.drop(columns = 'date',inplace = True)
df


# #### Scaling

# In[29]:


#min_max_scaler = MinMaxScaler()
scaler = StandardScaler(with_mean = False)
cols  = ['smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_workers']

df[cols] = scaler.fit_transform(df[cols])


# ### Train/Test/Val Split
# Split into X and Y, and then into Train, validation and test set.
# As we are dealing with timeseries data, we do not want to split the data at random, but the first 60% will be training data, 20% validation data and 20% test data, sorted by date. This way, we can simulate the fact that we predict on future data.

# In[30]:


df.sort_values(by = "date")
X, y = df.drop(['actual_productivity'], axis=1), df['actual_productivity']


# In[31]:


n = df.shape[0]
train_size = round(n * 0.6)
val_size = round(n*0.8)

X_train = X.iloc[:train_size,:]
y_train = y.iloc[:train_size]
X_val = X.iloc[train_size:val_size,:]
y_val =  y.iloc[train_size:val_size]
X_test = X.iloc[val_size:,:]
y_test =  y.iloc[val_size:]


# In[32]:


print('Shapes:')
print(f"Train: x={X_train.shape}, y={y_train.shape}")
print(f"Test: x={X_test.shape}, y={y_test.shape}")
print(f"Validation: x={X_val.shape}, y={y_val.shape}")


# ## Build models

# ### Setup of evaluation

# In[33]:


# Store results
performance_Results = pd.DataFrame(columns = ['models','mae','mse', 'rmse' ,'mape', 'R2'])

# Test models
def evaluate_model(model_name, Y_actual, Y_Predicted, df):
    # Calculate metrics
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    mae = metrics.mean_absolute_error(Y_actual, Y_Predicted)
    mse = metrics.mean_squared_error(Y_actual, Y_Predicted)
    rmse = np.sqrt(metrics.mean_squared_error(Y_actual, Y_Predicted))
    r2 = metrics.r2_score(Y_actual, Y_Predicted)
    
    # Concatenate
    df_temp = {'models':model_name,'mae':mae,'mse':mse, 'rmse':rmse, 'mape':mape, 'R2': r2}
    df = df.append(df_temp, ignore_index = True)
    
    # Visualize
    plt.scatter(Y_actual, Y_Predicted)
    plt.plot([Y_actual.min(), Y_actual.max()], [Y_actual.min(), Y_actual.max()], 'k--', lw=4)
    plt.xlabel("Actual Productivity")
    plt.ylabel("Predicted Productivity")
    plt.title(f"Actual vs Predicted for {model_name} Model")
    plt.savefig(GRAPHICS_LOCATION + 'Results_Actual_Vs_Predicted_Model_' + model_name.replace(" ", "_") + '.png')
    plt.show()
    
    # Show metrics
    print(f"MAPE:{mape}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE:{rmse}")
    print(f"R2:  {r2}")
    print("------------------------")
    
    # End
    return df


# ### Linear Regression

# In[34]:


# Build Model
model = LinearRegression()
# Fit Model
model.fit(X_train,y_train)
# Predict on test set
y_pred = model.predict(X_val)
# Evaluate model
performance_Results = evaluate_model('Linear Regression', y_val, y_pred, performance_Results)


# ### Lasso Regression

# In[35]:


# Build Model
model = Lasso()
# Fit Model
model.fit(X_train,y_train)
# Predict on test set
y_pred = model.predict(X_val)
# Evaluate model
performance_Results = evaluate_model('Lasso Regression', y_val, y_pred, performance_Results)


# ### Ridge Regression

# In[36]:


# Build Model
model = Ridge()
# Fit Model
model.fit(X_train,y_train)
# Predict on test set
y_pred = model.predict(X_val)
# Evaluate model
performance_Results = evaluate_model('Ridge Regression', y_val, y_pred, performance_Results)


# ### Random Forrest

# In[37]:


# Build Model
model = RandomForestRegressor(n_estimators = 100 ,  random_state = 10)
# Fit Model
model.fit(X_train,y_train)
# Predict on test set
performance_Results = evaluate_model('Linear Regression', y_val, y_pred, performance_Results)# Evaluate model
performance_Results = evaluate_model('Random Forrest Regression', y_val, y_pred, performance_Results)


# ### Support Vector Regression

# In[38]:


# Build Model
model = SVR(C=25)
# Fit Model
model.fit(X_train,y_train)
# Predict on test set
y_pred = model.predict(X_val)
# Evaluate model
performance_Results = evaluate_model('Support Vector Regression', y_val, y_pred, performance_Results)


# ### K Nearest Neighbours

# In[39]:


# Build Model
model = KNeighborsRegressor(n_neighbors=3)
# Fit Model
model.fit(X_train,y_train)
# Predict on test set
y_pred = model.predict(X_val)
# Evaluate model
performance_Results = evaluate_model('3 Nearest Neighbours', y_val, y_pred, performance_Results)


# ### Extreme Gradient Descent Boosting

# In[40]:


# Build Model
model = xgb.XGBRegressor(verbosity = 0)
# Fit Model
model.fit(X_train,y_train)
# Predict on test set
y_pred = model.predict(X_val)
# Evaluate model
performance_Results = evaluate_model('XGBRegression', y_val, y_pred, performance_Results)


# ### Gradient Boosting

# In[41]:


# Build Model
model = GradientBoostingRegressor(alpha=0.9,learning_rate=0.05, max_depth=2, min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=30)
# Fit Model
model.fit(X_train,y_train)
# Predict on test set
y_pred = model.predict(X_val)
# Evaluate model
performance_Results = evaluate_model('Gradient Boost', y_val, y_pred, performance_Results)


# ## Evaluate models

# In[42]:


performance_Results = evaluate_model('Target set', y_val, X_val["targeted_productivity"], performance_Results)


# In[43]:


def create_barplot(metric):
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=performance_Results
                     , x="models"
                     , y=metric)
    ax.set_title(str.upper(metric) + ' comparison between models')
    ax.set(xlabel=None)

    plt.xticks(rotation=90)
    plt.savefig(GRAPHICS_LOCATION + 'Results_Model_Comparisson_' + str.upper(metric) + '.png')
    plt.show()


# #### Table with comparisson of models

# In[44]:


performance_Results.style.background_gradient(axis=0)


# #### MAE 

# In[45]:


create_barplot('mae')


# #### MSE

# In[46]:


create_barplot('mse')


# #### RMSE

# In[47]:


create_barplot('rmse')


# #### MAPE

# In[48]:


create_barplot('mape')


# #### R2

# In[49]:


create_barplot('R2')


# **Intermediary Conclusion:** Random forrest seems to be the best results, and is also quite easy to make explainable. We therefore decide to use that for the final evaluation

# ### Hyperparameter search
# 
# Found at https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# In[50]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[51]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor(random_state = 10)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=10, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[52]:


rf_random.best_params_


# In[53]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    mae = np.mean(abs(predictions - test_labels))
    mse = np.mean((predictions - test_labels)**2)
    rmse = mse**(1/2)

    print('Model Performance')
    print('MAE: {:0.4f}'.format(mae))
    print('MSE = {:0.2f}'.format(mse))
    print('RMSE = {:0.2f}'.format(rmse))
    return mae

base_model = RandomForestRegressor(n_estimators = 100, random_state = 10)

base_model.fit(X_train, y_train)

base_performance = evaluate(base_model, X_val, y_val)

best_random = rf_random.best_estimator_
best_performance = evaluate(best_random, X_val, y_val)

print('Improvement of {:0.2f}%.'.format( 100 * (best_performance - base_performance) / base_performance))
# 


# In[ ]:





# # 5) Evaluation

# **a. Apply the final model** on the test data and document performance.

# In[54]:


rf_final_only_train = RandomForestRegressor( random_state = 10, n_estimators = 400, min_samples_split = 10, min_samples_leaf = 4, 
                              max_features = 'auto', max_depth = 70, bootstrap = True)

rf_final_only_train.fit(X_train, y_train)


# ## Retrain the model with identical hyperparameters using the full train and test set
# 

# ### Concatenate Train and Test

# In[55]:


X_final = pd.concat([X_train, X_val])
print(f"Concatenated X_train {X_train.shape} and X_test {X_val.shape}. Got X_final with shape {X_final.shape}")

y_final = pd.concat([y_train, y_val])
print(f"Concatenated X_train {y_train.shape} and X_test {y_val.shape}. Got X_final with shape {y_final.shape}")


# ### Rebuild model on full dataset

# In[56]:


# Build Model
rf_final_incl_val = RandomForestRegressor( random_state = 10, n_estimators = 400, min_samples_split = 10, min_samples_leaf = 4, 
                              max_features = 'auto', max_depth = 70, bootstrap = True)
# Fit Model
rf_final_incl_val.fit(X_final, y_final)


# ### compare performance on different training

# In[57]:


performance_Results = pd.DataFrame(columns = ['models','mae','mse', 'rmse' ,'mape', 'R2'])

pred_train = rf_final_only_train.predict(X_train)
pred_val = rf_final_only_train.predict(X_val)
pred_test = rf_final_only_train.predict(X_test)

performance_Results = evaluate_model('Final Model(training)', y_train, pred_train, performance_Results)
performance_Results = evaluate_model('Final Model(validation)', y_val, pred_val, performance_Results)
performance_Results = evaluate_model('Final Model(test)', y_test, pred_test, performance_Results)


# In[58]:


performance_Results


# In[59]:


performance_Results = pd.DataFrame(columns = ['models','mae','mse', 'rmse' ,'mape', 'R2'])
# predict for both models
train_only_pred = rf_final_only_train.predict(X_test)
incl_val_pred = rf_final_incl_val.predict(X_test)
performance_Results = evaluate_model('Final Model(training data)', y_test, train_only_pred, performance_Results)
performance_Results = evaluate_model('Final Model(training and val data)', y_test, incl_val_pred, performance_Results)
performance_Results


# In[60]:


performance_Results


# In[61]:


d = pd.melt(performance_Results, id_vars='models', value_vars=['mae', 'mse','rmse', 'R2'])


# In[62]:


plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")
ax = sns.barplot(data=d
                     , x="variable"
                     , y='value'
                    , hue = "models")
ax.set_title('Performances Comparison of different training sizes')
ax.set(xlabel=None)

plt.xticks(rotation=90)


# In[63]:


performance_Results.set_index("models", inplace = True)
performance_Results.style.background_gradient(axis=0)


# ## Explainability
# Find out which features were the biggest explainers for our prediction

# In[64]:


feature_list = list(X_final.columns.values)


# ### All features

# In[65]:


# Get numerical feature importances
importances = list(model.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out information
print("Variables with the biggest effect:")
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Visualize
feature_names = []
y_values = []
for a,b in feature_importances:
    feature_names.append(a)
    y_values.append(b)
    
x_values = list(range(len(feature_names)))
plt.bar(x_values, y_values, orientation = 'vertical')
plt.xticks(x_values, feature_names, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.savefig('output_graphics/featureImportance_all_final_model.png')
plt.show()


# In[66]:


var_importance = pd.DataFrame(feature_importances, columns = ["Variable", "Importance"])
display(var_importance.style.background_gradient(cmap="Blues"))


# ### Only the most important ones

# In[67]:


# Get numerical feature importances
importances = list(model.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)  if importance>= 0.02]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out information
print("Variables with the biggest effect:")
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Visualize
feature_names = []
y_values = []
for a,b in feature_importances:
    feature_names.append(a)
    y_values.append(b)
    
x_values = list(range(len(feature_names)))
plt.bar(x_values, y_values, orientation = 'vertical')
plt.xticks(x_values, feature_names, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.savefig('output_graphics/featureImportance_top_only_final_model.png')
plt.show()


# ## Run on test
# Should probably kill this - Totally overfitting!!
# Predict on test set
y_pred = model.predict(X_test)
# Evaluate model
performance_Results = evaluate_model('Random Forrest Regression Validation', y_test, y_pred, performance_Results)
# ## Run on validation 

# In[68]:


# Predict on validation
y_pred = model.predict(X_val)
# Evaluate model
performance_Results = evaluate_model('Validation Random Forrest Regression', y_val, y_pred, performance_Results)


# ## Analysis of performance in detail

# In[71]:


y_pred = pd.Series(y_pred)
y_pred.index = y_val.index


# In[72]:



### test on different teams separately
performance_Results = pd.DataFrame(columns = ['models','mae','mse', 'rmse' ,'mape', 'R2'])

for i in range(9):
    i = i+1
    performance_Results = evaluate_model('Team ' + str(i), y_val.loc[X_val["team_"+str(i)] == 1], y_pred.loc[X_val["team_"+str(i)] == 1], performance_Results)
    
performance_Results


# In[73]:


create_barplot("mae")


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing DF
import pandas as pd

demographics_path = '/Users/vincent/Desktop/Demographics.xlsx'
demographics_df = pd.read_excel(demographics_path)

demographics_df


# In[2]:


#Visualization of numeric columns data in whisker box plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

columns_to_convert = [
    'Age (in years)',
    'CAO Points (100 to 600)',
    'Daily travel to DCU (in km, 0 if on-campus)',
    'Average year 1 exam result (as %)',
    'Seat row in class'
]

for column in columns_to_convert:
    demographics_df[column] = pd.to_numeric(demographics_df[column], errors='coerce')

numeric_columns = demographics_df.select_dtypes(include=['number']).columns

n = len(numeric_columns)
ncols = 4  
nrows = n // ncols + (n % ncols > 0)

plt.figure(figsize=(ncols * 5, nrows * 5))  

for i, column in enumerate(numeric_columns, 1):
    plt.subplot(nrows, ncols, i)
    plt.boxplot(demographics_df[column].dropna(), patch_artist=True)  
    plt.title(column)
    plt.ylabel('Value')

plt.tight_layout()
plt.savefig('/Users/vincent/Desktop/As.1/Boxplots/dem_boxplots.png')  
plt.close()


# In[3]:


#create a correlation matrix for the df

import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

numeric_df = demographics_df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

plt.figure(figsize=(10, 8.8))

sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt=".2f",
            annot=True, annot_kws={"size": 8})

plt.title('Correlation Matrix')
plt.tight_layout()

png_path = '/Users/vincent/Desktop/As.1/CorrelationMatrix.png'

plt.savefig(png_path, bbox_inches='tight', dpi=600)

plt.show()


# In[4]:


#check for statistically sound correlations which can be used for regressions to full the NA values in columns of interest

import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd

columns_of_interest = [
    'Age (in years)',
    'CAO Points (100 to 600)',
    'Daily travel to DCU (in km, 0 if on-campus)',
    'Average year 1 exam result (as %)',
    'Seat row in class'
]

all_significant_correlations = {}

for column in columns_of_interest:
    correlations_with_p = []
    
    for target_column in numeric_df.columns:
        if column == target_column:
            continue  
        
        valid_data = demographics_df[[column, target_column]].dropna()
        
        if len(valid_data) >= 2:
            r, p_value = stats.pearsonr(valid_data[column], valid_data[target_column])
            if p_value < 0.05:
                correlations_with_p.append((target_column, r, p_value))
    
    correlations_with_p.sort(key=lambda x: abs(x[1]), reverse=True)
    
    all_significant_correlations[column] = correlations_with_p

for column, correlations in all_significant_correlations.items():
    print(f"Statistically significant correlations for '{column}':")
    
    variables_for_vif = [column] + [corr[0] for corr in correlations]
    df_vif = demographics_df[variables_for_vif].dropna()
    df_vif = add_constant(df_vif)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
    
    for target, r, p_value in correlations:
        vif = vif_data[vif_data['Variable'] == target]['VIF'].values[0] if target in vif_data['Variable'].values else 'N/A'
        print(f"  {target}: Correlation = {r:.3f}, P-value = {p_value:.3g}, VIF = {vif}")
    
    print("\n")


# In[5]:


#Prediction model for Age column, all values with statistical significance and no multicollinearity concern are used 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

predictors = [
    'Number of older siblings', 
    'Number of younger siblings', 
    'Seat row in class', 
    'Last 4 digits of your mobile (0000 to 9999)',
    'Daily travel to DCU (in km, 0 if on-campus)'
]

df_notnull = demographics_df.dropna(subset=['Age (in years)'] + predictors)

X = df_notnull[predictors]
y = df_notnull['Age (in years)']

model = LinearRegression()
model.fit(X, y)

df_null = demographics_df[demographics_df['Age (in years)'].isnull() & demographics_df[predictors].notnull().all(axis=1)]
X_null = df_null[predictors]

if not X_null.empty:
    predicted_age = model.predict(X_null)
    demographics_df.loc[demographics_df['Age (in years)'].isnull() & demographics_df[predictors].notnull().all(axis=1), 
    'Age (in years)'] = predicted_age

demographics_df.head()


# In[6]:


#Prediction model for CAO column

predictors_for_cao_points = ['Last 4 digits of your mobile (0000 to 9999)', 'Average year 1 exam result (as %)']

df_notnull = demographics_df.dropna(subset=['CAO Points (100 to 600)'] + predictors_for_cao_points)

X = df_notnull[predictors_for_cao_points]
y = df_notnull['CAO Points (100 to 600)']

model = LinearRegression()
model.fit(X, y)

df_null = demographics_df[demographics_df['CAO Points (100 to 600)'].isnull() & demographics_df[predictors_for_cao_points].notnull().all(axis=1)]
X_null = df_null[predictors_for_cao_points]

if not X_null.empty:
    predicted_cao_points = model.predict(X_null)
    demographics_df.loc[demographics_df['CAO Points (100 to 600)'].isnull() & demographics_df[predictors_for_cao_points].notnull().all(axis=1), 'CAO Points (100 to 600)'] = predicted_cao_points

demographics_df.head()


# In[7]:


#Prediction for Travel to DCU

predictors_for_daily_travel = ['Age (in years)']

df_notnull = demographics_df.dropna(subset=['Daily travel to DCU (in km, 0 if on-campus)'] + predictors_for_daily_travel)

X = df_notnull[predictors_for_daily_travel]
y = df_notnull['Daily travel to DCU (in km, 0 if on-campus)']

model = LinearRegression()
model.fit(X, y)

df_null = demographics_df[demographics_df['Daily travel to DCU (in km, 0 if on-campus)'].isnull() & demographics_df[predictors_for_daily_travel].notnull().all(axis=1)]
X_null = df_null[predictors_for_daily_travel]

if not X_null.empty:
    predicted_daily_travel = model.predict(X_null)
   
    demographics_df.loc[demographics_df['Daily travel to DCU (in km, 0 if on-campus)'].isnull() & demographics_df[predictors_for_daily_travel].notnull().all(axis=1), 'Daily travel to DCU (in km, 0 if on-campus)'] = predicted_daily_travel

demographics_df.head()


# In[8]:


#Prediction for year 1 results

predictors_for_avg_year_1_results = ['CAO Points (100 to 600)']

df_notnull = demographics_df.dropna(subset=['Average year 1 exam result (as %)'] + predictors_for_avg_year_1_results)

X = df_notnull[predictors_for_avg_year_1_results]
y = df_notnull['Average year 1 exam result (as %)']

model = LinearRegression()
model.fit(X, y)

df_null = demographics_df[demographics_df['Average year 1 exam result (as %)'].isnull() & demographics_df[predictors_for_avg_year_1_results].notnull().all(axis=1)]
X_null = df_null[predictors_for_avg_year_1_results]

if not X_null.empty:
    predicted_avg_year_1_results = model.predict(X_null)
    demographics_df.loc[demographics_df['Average year 1 exam result (as %)'].isnull() & demographics_df[predictors_for_avg_year_1_results].notnull().all(axis=1), 'Average year 1 exam result (as %)'] = predicted_avg_year_1_results

demographics_df.head()


# In[9]:


#Prediction for Seat row in class
predictors_for_seat_row = ['Age (in years)', 'Shoe size']

df_notnull = demographics_df.dropna(subset=['Seat row in class'] + predictors_for_seat_row)

X = df_notnull[predictors_for_seat_row]
y = df_notnull['Seat row in class']

model = LinearRegression()
model.fit(X, y)

df_null = demographics_df[demographics_df['Seat row in class'].isnull() & demographics_df[predictors_for_seat_row].notnull().all(axis=1)]
X_null = df_null[predictors_for_seat_row]

if not X_null.empty:
    predicted_seat_row = model.predict(X_null)
    demographics_df.loc[demographics_df['Seat row in class'].isnull() & demographics_df[predictors_for_seat_row].notnull().all(axis=1), 'Seat row in class'] = predicted_seat_row

demographics_df.head()


# In[10]:


# Load dataframe and identify columns to merge
personalities_path = '/Users/vincent/Desktop/Personalities.xlsx'
personalities_df = pd.read_excel(personalities_path)

print("Demographics DataFrame columns:", demographics_df.columns.tolist())
print("Personalities DataFrame columns:", personalities_df.columns.tolist())


# In[11]:


#Merging the dataframes and dropping the second phone number column
merged_df = pd.merge(demographics_df, personalities_df, 
                     left_on='Last 4 digits of your mobile (0000 to 9999)', 
                     right_on='Last 4 digits of your mobile (same as on previous form)',  
                     how='inner')
merged_df.drop(columns=['Last 4 digits of your mobile (same as on previous form)'], inplace=True)

merged_df


# In[12]:


merged_df.to_csv('/Users/vincent/Desktop/merged_data.csv', index=False)

merged_df.head()


# In[13]:


#describe the merged_df
file_path = '/Users/vincent/Desktop/merged_data.csv'
data = pd.read_csv(file_path)

description_stats = merged_df.describe()

median_series = merged_df.median(numeric_only=True)
median_df = pd.DataFrame(median_series).transpose()
median_df.index = ['median']

description_with_median = pd.concat([description_stats, median_df.reindex(columns=description_stats.columns)], axis=0)

description_with_median


# In[14]:


#Average age is approximately 21.6 years, with a wide age range (18 to 58 years), indicating both traditional and non-traditional students.
#Average CAO points score is around 501.2, with scores ranging from 130 to 578, showing a wide range of academic abilities.
#Students travel an average of 10.94 km, with a broad range from 0 to 103 km, highlighting varied commuting distances.
#The average Year 1 exam result is 68.53%, with results ranging from 42% to 95%, indicating diverse academic performance.
#Students have an average of 1.22 older and 0.87 younger siblings, suggesting insights into family sizes.
#With an average 'Old Dublin postcode' value of 5.12 and a maximum of 22, data indicates many students live outside central Dublin.
#Averages for height (175.44 cm), weight (71.20 kg), and shoe size (7.76) show physical diversity among students.
#Students provided mid-scale average ratings on traits like Extraversion, Intuition, etc., showing a balanced mix of personalities but with wide score ranges.
#Maximum values for age, daily travel, height, and weight significantly exceed the 75th percentile, suggesting outliers in the dataset.



# In[15]:


# visualize merged_df personality data using whisker box plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

columns_of_interest = [
    'Your rating for EXTRAVERSION (vs. introversion)',
    'Your rating for INTUITION (vs. observation)',
    'Your rating for THINKING (vs. feeling)',
    'Your rating for JUDGING (vs. prospecting)',
    'Your rating for ASSERTIVE (vs. turbulent)'
]

n = len(columns_of_interest)
ncols = 5
nrows = n // ncols + (n % ncols > 0)

plt.figure(figsize=(ncols * 5, nrows * 5))

for i, column in enumerate(columns_of_interest, 1):
    plt.subplot(nrows, ncols, i)
    plt.boxplot(merged_df[column].dropna(), patch_artist=True)
    plt.title(column)
    plt.ylabel('Value')

plt.tight_layout()  # Adjust subplots to fit in the figure area
plt.savefig('/Users/vincent/Desktop/As.1/Boxplots/merged_boxplots.png')
plt.close()


# In[16]:


# additional check whole merged_df with z-score and also get inices for potential replacement of outliers
from scipy.stats import zscore

numeric_df = merged_df.select_dtypes(include=[np.number])

z_scores = numeric_df.apply(zscore)

threshold = 3

outliers = (np.abs(z_scores) > threshold)

outlier_indices = np.where(outliers)

outlier_info = [(numeric_df.index[outlier_indices[0][i]], numeric_df.columns[outlier_indices[1][i]]) for i in range(len(outlier_indices[0]))]

for index, column in outlier_info:
    print(f"Outlier found in row {index}, column {column}")


# In[17]:


#check if these are wrong values or just true outliers
rows_to_copy = [12, 18, 47, 58, 59]

outliers_df = merged_df.loc[rows_to_copy].copy()

outliers_df


# In[18]:


# there are outliers in the personality data which are most likely wrong data entries, check if regressions are possible to 
# replace these values

columns_of_interest = [
    'Your rating for EXTRAVERSION (vs. introversion)',
    'Your rating for INTUITION (vs. observation)',
    'Your rating for THINKING (vs. feeling)',
    'Your rating for JUDGING (vs. prospecting)',
    'Your rating for ASSERTIVE (vs. turbulent)'
]

all_significant_correlations = {}

for column in columns_of_interest:
    correlations_with_p = []
    
    for target_column in numeric_df.columns:
        if column == target_column:
            continue 
        
        valid_data = merged_df[[column, target_column]].dropna()
        
        if len(valid_data) >= 2:
            r, p_value = stats.pearsonr(valid_data[column], valid_data[target_column])
            if p_value < 0.05:
                correlations_with_p.append((target_column, r, p_value)) 
    
    correlations_with_p.sort(key=lambda x: abs(x[1]), reverse=True)
    
    all_significant_correlations[column] = correlations_with_p

for column, correlations in all_significant_correlations.items():
    print(f"Statistically significant correlations for '{column}':")
    
    variables_for_vif = [column] + [corr[0] for corr in correlations]
    df_vif = merged_df[variables_for_vif].dropna()
    df_vif = add_constant(df_vif)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
    
    for target, r, p_value in correlations:
        vif = vif_data[vif_data['Variable'] == target]['VIF'].values[0] if target in vif_data['Variable'].values else 'N/A'
        print(f"  {target}: Correlation = {r:.3f}, P-value = {p_value:.3g}, VIF = {vif}")
    
    print("\n")


# In[19]:


#z-score for outlier detection and return the index of the row 

from scipy.stats import zscore

columns_of_interest = [
    'Your rating for EXTRAVERSION (vs. introversion)',
    'Your rating for INTUITION (vs. observation)',
    'Your rating for THINKING (vs. feeling)',
    'Your rating for JUDGING (vs. prospecting)',
    'Your rating for ASSERTIVE (vs. turbulent)'
]

outlier_mask = (np.abs(zscore(merged_df[columns_of_interest])) > 3).any(axis=1)

outliers_index = merged_df[outlier_mask].index

print(outliers_index)


# In[20]:


#manually replacing outliers asuming an input error and not having sufficient data to replace values with a regression

row_index = 18  

columns_with_outliers = [
    'Your rating for EXTRAVERSION (vs. introversion)',
    'Your rating for INTUITION (vs. observation)',
    'Your rating for THINKING (vs. feeling)',
    'Your rating for JUDGING (vs. prospecting)',
    'Your rating for ASSERTIVE (vs. turbulent)'
]

new_values = [60, 70, 70, 70, 60]

for column, new_value in zip(columns_with_outliers, new_values):
    merged_df.loc[row_index, column] = new_value

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)  
merged_df


# In[21]:


#describe the merged_df with the replaced outliers
file_path = '/Users/vincent/Desktop/merged_data.csv'
data = pd.read_csv(file_path)

description_stats = merged_df.describe()

median_series = merged_df.median(numeric_only=True)
median_df = pd.DataFrame(median_series).transpose()
median_df.index = ['median']

description_with_median = pd.concat([description_stats, median_df.reindex(columns=description_stats.columns)], axis=0)

import dataframe_image as dfi
output_image_path = '/Users/vincent/Desktop/description_with_median.png'
dfi.export(description_with_median, output_image_path)

print(f"Exported DataFrame to PNG at '{output_image_path}'")

description_with_median


# In[22]:


################################################################################################################

# Analyse the cleaned and merged_df

################################################################################################################


# In[23]:


# encode the nominal data and check for correlations

categorical_columns_nominal = ['Hair colour', 'Star sign', 'Eye colour', 'Gender']

data_encoded = pd.get_dummies(data, columns=categorical_columns_nominal, drop_first=False)

correlation_matrix = data_encoded.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

plt.figure(figsize=(20, 20))

sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt=".2f",
            annot=True, annot_kws={"size": 8})

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.title('Full Correlation Matrix Heatmap')

png_file_path = '/Users/vincent/Desktop/As.1/FullCorrelationMatrix.png'

plt.savefig(png_path, bbox_inches='tight', dpi=600)

plt.show()


# In[24]:


# 5 highest correlations of numerics (bei categorical bekommen wir komische ergenisse wegen der dummies)

numeric_data = data.select_dtypes(include=[np.number])

correlation_matrix = numeric_data.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

masked_correlation_matrix = correlation_matrix.mask(mask)

flat_correlations = masked_correlation_matrix.unstack().dropna().abs()

sorted_correlations = flat_correlations.sort_values(ascending=False)

print(sorted_correlations.head(5))


# In[39]:


#create scatter plots for the 5 highest correlations of numeric columns

import seaborn as sns
import matplotlib.pyplot as plt
import os

variable_pairs_high = [
    ('Height (in cm)', 'Weight (in kg)'),
    ('Height (in cm)', 'Shoe size'),
    ('Weight (in kg)', 'Shoe size'),
    ('Your rating for THINKING (vs. feeling)', 'Your rating for ASSERTIVE (vs. turbulent)'),
    ('Your rating for EXTRAVERSION (vs. introversion)', 'Your rating for INTUITION (vs. observation)'),
    ('Your rating for ASSERTIVE (vs. turbulent)', 'Average year 1 exam result (as %)'),
]

output_directory = '/Users/vincent/Desktop/As.1/RegressionLines'

os.makedirs(output_directory, exist_ok=True)

for var1, var2 in variable_pairs_high:
    sns.regplot(x=numeric_data[var1], y=numeric_data[var2], line_kws={"color": "red"})
    plt.title(f'Scatter Plot of {var1} vs {var2}')
    plt.xlabel(var1)
    plt.ylabel(var2)
    
    filename = f'{var1}_vs_{var2}_regression_lines.png'.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
    png_path = os.path.join(output_directory, filename)
    
    plt.savefig(png_path, bbox_inches='tight', dpi=600)
    plt.show()
    
    plt.close()

#regression line within 95% confidence interval


# In[26]:


#check statistic significance of 5 highest correlations with p value 

from scipy.stats import pearsonr

for var1, var2 in variable_pairs_high:
    correlation_coefficient, p_value = pearsonr(numeric_data[var1], numeric_data[var2])
    
    
    print(f"Correlation between {var1} and {var2}:")
    print(f"  Correlation coefficient: {correlation_coefficient:.6f}")
    print(f"  P-value: {p_value:.6e}")
    print() 
#high statistic significant based on p value


# In[27]:


# checking for inversive relationship 
numeric_data = data.select_dtypes(include=[np.number])

correlation_matrix = numeric_data.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

masked_correlation_matrix = correlation_matrix.mask(mask)

# Instead of dropping the negative correlations, sort the flattened correlations by their actual values, not their absolute values
flat_correlations = masked_correlation_matrix.unstack().dropna()

sorted_correlations = flat_correlations.sort_values()

print(sorted_correlations.head(5))


# In[28]:


from scipy.stats import pearsonr

variable_pairs_low = [
    ('Number of older siblings', 'Number of younger siblings'),
    ('Weight (in kg)', 'Your rating for INTUITION (vs. observation)'),
    ('Height (in cm)', 'Your rating for INTUITION (vs. observation)'),
    ('Old Dublin postcode (0 if outside Dublin)', 'Last 4 digits of your mobile (0000 to 9999)'),
    ('Age (in years)', 'Your rating for EXTRAVERSION (vs. introversion)')
]

for var1, var2 in variable_pairs_low:   
    series1 = numeric_data[var1]
    series2 = numeric_data[var2]
    
    
    corr_coefficient, p_value = pearsonr(series1, series2)
    
    print(f"Correlation and p-value for {var1} and {var2}:")
    print(f"  Correlation Coefficient: {corr_coefficient:.4f}")
    print(f"  P-value: {p_value:.4e}")
    print() 

#unliky to be coincidental, not as unlikly as high values


# In[29]:


# data visualisation for numerics distribution

plt.rcParams.update({'font.size': 8})

data.hist(bins=15, figsize=(15, 10), layout=(5, 4));

output_directory = '/Users/vincent/Desktop/As.1/Numerics_Distribution'

os.makedirs(output_directory, exist_ok=True)

file_path = os.path.join(output_directory, f'{column}_numerics_distribution.png')

plt.savefig(file_path, dpi=300)  
plt.close()


# In[30]:


#regression for personality traits 
import statsmodels.api as sm
X = numeric_data[['Age (in years)', 'CAO Points (100 to 600)', 'Average year 1 exam result (as %)',
                  'Daily travel to DCU (in km, 0 if on-campus)', 'Seat row in class', 'Number of older siblings',
                  'Number of younger siblings', 'Old Dublin postcode (0 if outside Dublin)', 'Height (in cm)',
                  'Weight (in kg)', 'Last 4 digits of your mobile (0000 to 9999)', 'Shoe size']]
X = sm.add_constant(X) 

personality_traits = [
    'Your rating for EXTRAVERSION (vs. introversion)',
    'Your rating for INTUITION (vs. observation)',
    'Your rating for THINKING (vs. feeling)',
    'Your rating for JUDGING (vs. prospecting)',
    'Your rating for ASSERTIVE (vs. turbulent)'
]
for trait in personality_traits:
    Y = numeric_data[trait]

    model = sm.OLS(Y, X).fit()
    
    print(f"Regression Analysis for: {trait}")
    print(model.summary())
    print("\n\n---\n\n")
    
    
#Taller individuals tend to have higher ratings for thinking (vs. feeling) (Coef = 1.0569, p-value = 0.013)
#Higher academic performance in the first year is associated with higher ratings for assertiveness (Coef = 0.6979, p-value = 0.011)
#Number of younger siblings positively influences intuition ratings (Coef = 2.7806, p-value = 0.248)
#Having more older siblings shows a positive trend with higher extraversion ratings (Coef = 3.4439, p-value = 0.188)
#Number of younger siblings doesn't significantly influence judging (vs. prospecting) ratings (Coef = 0.1020, p-value = 0.972)
#Weight has a negative trend with assertiveness ratings (Coef = -0.6095, p-value = 0.133)
#usw.


# In[31]:


#plot relationship between height and thinking vs. feeling

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
sns.regplot(x='Height (in cm)', y='Your rating for THINKING (vs. feeling)', data=numeric_data)
plt.title('Relationship Between Height and Thinking vs. Feeling Ratings')
plt.xlabel('Height (in cm)')
plt.ylabel('Your rating for THINKING (vs. feeling)')

plt.show()


# In[32]:


# ellbow method to determin optimal k for PCA 

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

numeric_data = data.select_dtypes(include=[float, int])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

sse = [] 
for k in range(1, 11): 
    kmeans = KMeans(n_clusters=k,n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[33]:


# KMeans clustering with 4 clusters for merged_df

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
numeric_data = merged_df[numeric_cols]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

k = 4
kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_data)

merged_df['Cluster'] = clusters

merged_df.head()


# In[34]:


# PCA visualization and checking how many components are needed

from sklearn.decomposition import PCA

numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
numeric_data = merged_df[numeric_cols]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

pca_full = PCA()
pca_full_result = pca_full.fit_transform(scaled_data)  

fig = plt.figure(figsize=(16, 6))

ax = fig.add_subplot(121, projection='3d')  
scatter = ax.scatter(pca_full_result[:, 0], pca_full_result[:, 1], pca_full_result[:, 2], c=merged_df['Cluster'], cmap='viridis', s=50, alpha=0.6)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('Cluster Visualization in 3D PCA Space')
cbar = plt.colorbar(scatter, shrink=0.5, pad=0.06, aspect=5)  
cbar.set_label('Cluster')

ax2 = fig.add_subplot(122)
ax2.plot(np.cumsum(pca_full.explained_variance_ratio_))
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Explained Variance by PCA Components')
ax2.grid(True)

plt.tight_layout()
plt.show()


# In[35]:


#Description of all components 
print("Explained variance ratio by principal component:")
print(pca_full.explained_variance_ratio_)
total_variance_explained = sum(pca_full.explained_variance_ratio_)
print(f"Total variance explained by all selected components: {total_variance_explained:.2f}")

print("\nContribution of features to principal components:")
features = numeric_cols
for i, component in enumerate(pca_full.components_):
    component_contributions = dict(zip(features, component))
    sorted_contributions = sorted(component_contributions.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPrincipal Component {i+1}:")
    for feature, contribution in sorted_contributions:
        print(f"{feature}: {contribution:.4f}")


# In[36]:


#7 component PCA to explain most of the variance in merged_df

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
numeric_data = data[numeric_cols]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

pca_7 = PCA(n_components=7)
pca_7_result = pca_7.fit_transform(scaled_data)


# In[37]:


#Description of the 7 components 
print("Explained variance ratio by principal component:")
print(pca_7.explained_variance_ratio_)
total_variance_explained = sum(pca_7.explained_variance_ratio_)
print(f"Total variance explained by all selected components: {total_variance_explained:.2f}")

print("\nContribution of features to principal components:")
features = numeric_cols
for i, component in enumerate(pca_7.components_):
    component_contributions = dict(zip(features, component))
    sorted_contributions = sorted(component_contributions.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPrincipal Component {i+1}:")
    for feature, contribution in sorted_contributions:
        print(f"{feature}: {contribution:.4f}")


# In[38]:


#7 components can explain about 72% of  the variance in this data set


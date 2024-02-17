import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# SETUP

# convert .csv to pandas dataframe
df = pd.read_csv("rocketfuel_data-2-1-1.csv");
# ensure .csv is read properly
print(df.head())

# QUESTION 1
# run linear regression
X = sm.add_constant(df['test']) # constant term for the intercept
y = df['tot_impr']
model = sm.OLS(y, X).fit()

# Print regression results
print('q1 lin reg', model.summary())

# QUESTION 2
# calculate conversion rate for each group
conversion_rates = df.groupby('test')['converted'].mean()
print('CVRs', conversion_rates)

# run linear regression
X = sm.add_constant(df['test']) # constant term for the intercept
y = df['converted']
model = sm.OLS(y, X).fit()

# Print regression results
print('q2 lin reg', model.summary())

# QUESTION 3
# calculate the total number of users involved in the experiment (length of sheet)
num_rows = len(df)
print('user_count', num_rows)

# calculate the cost of the test
num_impressions = sum(df['tot_impr'])
cost_to_advertise = (num_impressions / 1000) * 9
print('cost of ads', cost_to_advertise)

# calculate the opportunity cost of using a control group
cvr_diff = 0.025547 - 0.017854
control_mem_count = len(df[df['test'] == 0])
opp_cost = cvr_diff * control_mem_count * 40
print('opp cost', opp_cost)

# QUESTION 4

# Define the bin edges
bin_edges = list(range(0, 201, 10)) + [float('inf')]

# Create bins for 'tot_impr' column
df['tot_impr_bins'] = pd.cut(df['tot_impr'], bins=bin_edges, right=False)

# Group by 'tot_impr_bins' and 'test' column and calculate conversion rates
conversion_rates = df.groupby(['tot_impr_bins', 'test'])['converted'].mean().unstack()

# Plotting
conversion_rates.plot(kind='bar', figsize=(10, 6))
plt.title('Conversion Rates by Total Impressions and Test/Control Group')
plt.xlabel('Total Impressions Bins')
plt.ylabel('Conversion Rate')
plt.xticks(rotation=45)
plt.legend(title='Group')
plt.show()

# Run a regression between the total_impressions and conversions

X = sm.add_constant(df['tot_impr'])
# Dependent variable
y = df['converted']

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print regression summary
print('impressions rel calc', model.summary())




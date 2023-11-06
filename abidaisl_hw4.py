#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install --upgrade seaborn')
get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# In[5]:


import pandas as pd
##########problem 1
data = pd.read_csv("/Users/abidaislam/IE459/scatter_data.csv")

print(data)


# In[6]:


plt.figure()
plt.plot(data['% x'], data[' y '], 'g^',linestyle='None', label='datapoints')
plt.title("Random dataset")
plt.ylabel('y axis')
plt.xlabel("x axis")
left_most_x = data['% x'].min()
right_most_x = data['% x'].max()

left_most_y = data[data['% x'] == left_most_x][' y '].min()
right_most_y = data[data['% x'] == right_most_x][' y '].min()

plt.plot([left_most_x, right_most_x], [left_most_y, right_most_y],
         color='r', linestyle='--', label='Red Dashed Line')
plt.legend()


# In[7]:


import numpy as np
data_2 = pd.read_csv("/Users/abidaislam/IE459/student_grades.csv")


print(data_2)
print(data_2.columns)


# In[8]:


###problem 2
plt.figure()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
bins = [0, 60, 70, 80, 90, 100]
labels = ['F', 'D', 'C', 'B', 'A']

data_2['Grade'] = pd.cut(data_2[' avgScore '], bins=bins, labels=labels, right=False)

hist, _ = np.histogram(data_2[' avgScore '], bins=bins)
plt.bar(labels, hist, color='orange', edgecolor='black', alpha=0.75)
###categorize the values in the 'avgScore' column into discrete bins specified by the bins parameter. The labels parameter provides labels for the bins, 
#and the right=False argument means that the intervals are left-closed and right-open.

for x, y in zip(labels, hist):
    plt.text(x, y + 0.1, str(y), ha='center')


plt.xlabel('Grade')
plt.ylabel('Number of students')
plt.title('Letter grades of students')
plt.show()


# In[12]:


#problem 3
data_5 = pd.read_csv("/Users/abidaislam/IE459/solution_data.csv")


print(data_5)


# In[16]:


optimal_values = data_5.loc[data_5['SolnMethod'] == 'optimal', ['% Problem', 'Value']].set_index('% Problem')['Value']
print(optimal_values)


# In[17]:


data_5['Optimality Gap'] = data_5.apply(lambda row: ((optimal_values[row['% Problem']] - row['Value']) / optimal_values[row['% Problem']]) * 100, axis=1)


# In[18]:


print(data_5)


# In[30]:


average_optimality_gap = data_5[data_5['SolnMethod'] != 'optimal'].groupby('SolnMethod')['Optimality Gap'].mean()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Bar plot(left subplot)
ax1.bar(average_optimality_gap.index, average_optimality_gap.values, color='orange', edgecolor='black')
ax1.set_xlabel('Heuristic Method')
ax1.set_ylabel('Optimality Gap %')
ax1.set_title('Mean Gaps')
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylim(0, 70)  
# Box plot (right subplot)
optimality_gaps = [data_5[data_5['SolnMethod'] == method]['Optimality Gap'].values for method in average_optimality_gap.index]

ax2.boxplot(optimality_gaps, labels=average_optimality_gap.index, showfliers=False, boxprops=dict(facecolor='white'),medianprops={'color':'orange'}, patch_artist=True)
ax2.set_xlabel('Heuristic Method')
ax2.set_ylabel('Numbers')
ax2.set_title('Distribution of gaps')
ax2.tick_params(axis='x', rotation=45)

ax2.set_ylim(0, 70)
plt.suptitle('Comparison of Optimality Gaps for Heuristics', y=1.05)
plt.show()


# In[31]:


##part 2
get_ipython().system('pip install --upgrade seaborn')
get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# In[32]:


plt.figure()
#link of the dataset: https://seaborn.pydata.org/examples/wide_form_violinplot.html
#link of the data:https://www.kaggle.com/datasets/uciml/iris
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
#Sets the default Seaborn theme with a white background and grid lines. 
#The style="whitegrid" argument specifies the style of the plot.

df = pd.read_csv("/Users/abidaislam/IE459/IRIS.csv")
palette = {"Iris-setosa": "b", "Iris-versicolor": "g", "Iris-virginica": "r"}

sns.violinplot(x="species", y='petal_length', data=df,hue="species",palette=palette)
#Creates a violin plot using Seaborn. 
##It visualizes the distribution of petal_length for each species category in the df. 
###The size=6 parameter adjusts the size of the plot elements.

g = sns.FacetGrid(df, hue="species")
##visualizing the distribution of a variable or the relationship between multiple 
##variables conditioned on one or more categorical variables (in this case, species).
g.map(sns.kdeplot, "petal_length")
##It shows the distribution of petal_length for each species category. 
##KDE is a way to estimate the probability density function of a continuous random variable.

# Customize the appearance of the plot
g.set(xlim=(0, 10), ylim=(0, 3), xlabel="Petal Length", ylabel="Density")
g.set(title="KDE Plot of Petal Length by Species")
g.add_legend()

##Sets the x-axis and y-axis limits, x-axis label, and y-axis label, adding the legend,


plt.tight_layout()

#Adjusts the spacing between subplots for a better layout.
plt.show()


# In[ ]:





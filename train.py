# Load libraries

from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")
pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore", category=FutureWarning)
LogisticRegression(solver='lbfgs')

# load in data
insulin_df1 = pd.read_csv('InsulinData.csv', low_memory=False )
insulin_df1 = insulin_df1[['Index', 'Date', 'Time', 'BWZ Carb Input (grams)']]
insulin_df2 = pd.read_csv('InsulinAndMealIntake670GPatient3.csv', low_memory=False)
insulin_df2 = insulin_df2[['Index', 'Date', 'Time', 'BWZ Carb Input (grams)']]
cgm_df1 = pd.read_csv('CGMData.csv', low_memory=False, )
cgm_df1 =cgm_df1[['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)']]
cgm_df2 = pd.read_csv('CGMData670GPatient3.csv', low_memory=False)
cgm_df2= cgm_df2[['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)']]


# manipulating data types
cgm_df1['DateTime'] = pd.to_datetime((cgm_df1['Date'] + ' ' + cgm_df1['Time']))
cgm_df1['Time'] = pd.to_datetime(cgm_df1['Time'], format="%H:%M:%S").dt.time

cgm_df2['DateTime'] = pd.to_datetime((cgm_df2['Date'] + ' ' + cgm_df2['Time']))
cgm_df2['Time'] = pd.to_datetime(cgm_df2['Time'], format="%H:%M:%S").dt.time

insulin_df1['DateTime'] = pd.to_datetime((insulin_df1['Date'] + ' ' + insulin_df1['Time']), format="%m/%d/%Y %H:%M:%S")
insulin_df1['Time'] = pd.to_datetime(insulin_df1['Time'], format="%H:%M:%S").dt.time

insulin_df2['DateTime'] = pd.to_datetime((insulin_df2['Date'] + ' ' + insulin_df2['Time']), format="%m/%d/%Y %H:%M:%S")
insulin_df2['Time'] = pd.to_datetime(insulin_df2['Time'], format="%H:%M:%S").dt.time

# get indices for the meal data
meal1_data = insulin_df1.loc[(insulin_df1['BWZ Carb Input (grams)'].isna() == False) & 
                         (insulin_df1['BWZ Carb Input (grams)'] != 0)][['Index', 'DateTime']]
meal2_data = insulin_df2.loc[(insulin_df2['BWZ Carb Input (grams)'].isna() == False) & 
                         (insulin_df2['BWZ Carb Input (grams)'] != 0)][['Index', 'DateTime']]

# get differences in meal time                        
meal1_data['DiffInMeals'] = meal1_data['DateTime'].diff(-1).to_frame()
meal2_data['DiffInMeals'] = meal2_data['DateTime'].diff(-1).to_frame()
meal1_start_time = meal1_data.loc[meal1_data['DiffInMeals'] > '02:00:00']
meal2_start_time = meal2_data.loc[meal2_data['DiffInMeals'] > '02:00:00']
noMeal1_start_time = meal1_data.loc[meal1_data['DiffInMeals'] >= '04:00:00']
noMeal2_start_time = meal2_data.loc[meal2_data['DiffInMeals'] >= '04:00:00']

# extract the meal time from all data
meal1_data_cols = list(range(-30, 120, 5))
meal_data1 = pd.DataFrame(columns=meal1_data_cols, index=range(len(meal1_start_time)))
for i in range(len(meal1_start_time) - 2):
    mealStartTemp = cgm_df1.loc[cgm_df1['DateTime'] >= meal1_start_time.iloc[i]['DateTime']]
    mealStartIndex = len(mealStartTemp) - 1
    preMealStartIndex = mealStartIndex + 6
    postMealStartIndex = mealStartIndex - 23
    meal_i = cgm_df1.iloc[postMealStartIndex:preMealStartIndex + 1, :]['Sensor Glucose (mg/dL)'].iloc[::-1].to_numpy()
    meal_data1.iloc[i] = meal_i
meal_data1.dropna(inplace=True)

meal_data_df2 = pd.DataFrame(columns=meal1_data_cols, index=range(len(meal2_start_time)))
for i in range(len(meal2_start_time) - 2):
    mealStartTemp = cgm_df2.loc[cgm_df2['DateTime'] >= meal2_start_time.iloc[i]['DateTime']]
    mealStartIndex = len(mealStartTemp) - 1
    preMealStartIndex = mealStartIndex + 6
    postMealStartIndex = mealStartIndex - 23
    meal_i = cgm_df2.iloc[postMealStartIndex:preMealStartIndex + 1, :]['Sensor Glucose (mg/dL)'].iloc[::-1].to_numpy()
    meal_data_df2.iloc[i] = meal_i
meal_data_df2.dropna(inplace=True)
meal_data_df = meal_data1.append(meal_data_df2)
meal_data_df.reset_index(inplace=True)
meal_data_df = meal_data_df.drop(['index'], axis=1)

noMeal_data1_cols = list(range(0, 120, 5))
nomeal_data1_df = pd.DataFrame(columns=noMeal_data1_cols, index=range(len(noMeal1_start_time)))
for i in range(len(noMeal1_start_time) - 2):
    noMealStart_Temp = cgm_df1.loc[cgm_df1['DateTime'] >= noMeal1_start_time.iloc[i]['DateTime']]
    index_noMeal_start = len(noMealStart_Temp) - 1
    index_noMeal_after = index_noMeal_start - 23
    noMeal_i = cgm_df1.iloc[index_noMeal_after:index_noMeal_start + 1, :]['Sensor Glucose (mg/dL)'].iloc[::-1].to_numpy()
    nomeal_data1_df.iloc[i] = noMeal_i
nomeal_data1_df.dropna(inplace=True)

noMeal_data2_df = pd.DataFrame(columns=noMeal_data1_cols, index=range(len(noMeal2_start_time)))
for i in range(len(noMeal2_start_time) - 2):
    noMealStart_Temp = cgm_df2.loc[cgm_df2['DateTime'] >= noMeal2_start_time.iloc[i]['DateTime']]
    index_noMeal_start = len(noMealStart_Temp) - 1
    index_noMeal_after = index_noMeal_start - 23
    noMeal_i = cgm_df2.iloc[index_noMeal_after:index_noMeal_start + 1, :]['Sensor Glucose (mg/dL)'].iloc[::-1].to_numpy()
    noMeal_data2_df.iloc[i] = noMeal_i

noMeal_data2_df.dropna(inplace=True)
noMeal_data_df = nomeal_data1_df.append(noMeal_data2_df)
noMeal_data_df.reset_index(inplace=True)
noMeal_data_df = noMeal_data_df.drop(['index'], axis=1)

# feature 1: max-min time
tmax_tmin = [] #time difference between max and mean time
max_min = []   #
derivative = []
rolling_MEAN = []  #rolling mean
window_MEAN = []   #window mean
labels = []  #data labels

meal_window1 = meal_data_df[[-30, -25, -20, -15, -10, -5]]
meal_window2 = meal_data_df[[0, 5, 10, 15, 20, 25]]
meal_window3 = meal_data_df[[30, 35, 40, 45, 50, 55]]
meal_window4 = meal_data_df[[60, 65, 70, 75, 80, 85]]
meal_window5 = meal_data_df[[90, 95, 100, 105, 110, 115]]

for i in range(len(meal_data_df)):
    # feature 1 - difference in cgm max and min time
    mealMaxIndex = meal_data_df.iloc[i].loc[meal_data_df.iloc[i] == meal_data_df.iloc[i].max()].index.tolist()[0]
    mealMinIndex = meal_data_df.iloc[i].loc[meal_data_df.iloc[i] == meal_data_df.iloc[i].min()].index.tolist()[0]
    tmax_tmin.append(mealMaxIndex - mealMinIndex)

    # feature 2 - difference in cgm max and min
    meal_diff_cgm = meal_data_df.iloc[i].max() - meal_data_df.iloc[i].min()
    max_min.append(meal_diff_cgm)

    # feature 3 - max of derivative cgm
    derivative.append((meal_data_df.iloc[i].diff() / 5).max())

    # feature 4 - rolling mean
    rolling_MEAN.append(meal_data_df.iloc[i].rolling(window=5).mean().mean())

    # feature 5 - window mean
    window_MEAN.append((meal_window1.iloc[i].mean() + meal_window2.iloc[i].mean() + meal_window3.iloc[i].mean() + 
                        meal_window4.iloc[i].mean() + meal_window5.iloc[i].mean()) / 5)

    # add class labels
    labels.append(1)

nomeal_tm_window1 = noMeal_data_df[[0, 5, 10, 15, 20, 25]]
nomeal_tm_window2 = noMeal_data_df[[30, 35, 40, 45, 50, 55]]
nomeal_tm_window3 = noMeal_data_df[[60, 65, 70, 75, 80, 85]]
nomeal_tm_window4 = noMeal_data_df[[90, 95, 100, 105, 110, 115]]

# find difference in cgm max and min time
for i in range(len(noMeal_data_df)):
    # feature 1 
    noMealMaxIndex = noMeal_data_df.iloc[i].loc[noMeal_data_df.iloc[i] == noMeal_data_df.iloc[i].max()].index.tolist()[0]
    noMealMinIndex = noMeal_data_df.iloc[i].loc[noMeal_data_df.iloc[i] == noMeal_data_df.iloc[i].min()].index.tolist()[0]
    tmax_tmin.append(noMealMaxIndex - noMealMinIndex)

    # feature 2 
    nomeal_diff_cgm = noMeal_data_df.iloc[i].max() - noMeal_data_df.iloc[i].min()
    max_min.append(nomeal_diff_cgm)

    # feature 3 - get the max of derivative cgm
    derivative.append((noMeal_data_df.iloc[i].diff() / 5).max())

    # feature 4 - compute the rolling mean
    rolling_MEAN.append(noMeal_data_df.iloc[i].rolling(window=5).mean().mean())

    # feature 5 - compute the window mean
    window_MEAN.append((nomeal_tm_window1.iloc[i].mean() 
                        +nomeal_tm_window2.iloc[i].mean() 
                        +nomeal_tm_window3.iloc[i].mean() 
                        +nomeal_tm_window4.iloc[i].mean()) / 5)

    labels.append(0)

# create feature matrix
f_matrix = pd.DataFrame(columns=[1, 2, 3, 4, 5, 'class'], index=range(len(meal_data_df) + len(noMeal_data_df)))
f_matrix[1] = tmax_tmin
f_matrix[2] = max_min
f_matrix[3] = derivative
f_matrix[4] = rolling_MEAN
f_matrix[5] = window_MEAN
f_matrix['class'] = labels

#split the data into training and test data
training_X, test_X, training_y, testing_X = train_test_split(f_matrix[[1, 2, 3, 4, 5]],
                                                    f_matrix[['class']],
                                                    test_size=0.33,
                                                   random_state=42)

#initiate scaler
scaler = StandardScaler()
training_X = scaler.fit_transform(training_X)
test_X = scaler.transform(test_X)

#Create a SVC classifier
the_classifier = svm.SVC()

## Train the classifier
the_classifier.fit(training_X, training_y)

#predict using the test data
pred_classifier = the_classifier.predict(test_X)

#write the generate machine to a pickle output file
out = open('pickle_model.pkl', 'wb')
pickle.dump(the_classifier, out)
out.close()


print("\n\n**************************END****************************")

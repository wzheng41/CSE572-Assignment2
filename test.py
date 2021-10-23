import pandas as pd
import pickle
import pickle_compat
pickle_compat.patch()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler

# load the test data
test_data_df = pd.read_csv('test.csv', header=None)
test_data_df.columns = range(0, 120, 5)

# load in the classifier
the_classifier = pickle.load(open('pickle_model.pkl', 'rb'))
the_scaler = StandardScaler()
print(the_classifier)

# feature extraction
tmax_tmin = []
max_min = []
derivative = []
rolling_MEAN = []
window_MEAN = []
labels = []

window1_df = test_data_df[[0, 5, 10, 15, 20, 25]]
window2_df = test_data_df[[30, 35, 40, 45, 50, 55]]
window3_df = test_data_df[[60, 65, 70, 75, 80, 85]]
window4_df = test_data_df[[90, 95, 100, 105, 110, 115]]

for i in range(len(test_data_df)):
    # get the difference in max and min time
    testMaxIndex = test_data_df.iloc[i].loc[test_data_df.iloc[i] == test_data_df.iloc[i].max()].index.tolist()[0]
    testMinIndex = test_data_df.iloc[i].loc[test_data_df.iloc[i] == test_data_df.iloc[i].min()].index.tolist()[0]
    tmax_tmin.append(testMaxIndex - testMinIndex)

    # compute difference in cgm max and min
    test_diff_cgm = test_data_df.iloc[i].max() - test_data_df.iloc[i].min()
    max_min.append(test_diff_cgm)

    # get the max derivative test data
    derivative.append((test_data_df.iloc[i].diff() / 5).max())

    # compute the rolling mean
    rolling_MEAN.append(test_data_df.iloc[i].rolling(window=5).mean().mean())

    # feature 5 - window mean
    window_MEAN.append((window1_df.iloc[i].mean() + window2_df.iloc[i].mean() + window3_df.iloc[i].mean() + 
                        window4_df.iloc[i].mean()) / 5)

f_matrix = pd.DataFrame(columns=[1, 2, 3, 4, 5], index=range(len(test_data_df)))
f_matrix[1] = tmax_tmin
f_matrix[2] = max_min
f_matrix[3] = derivative
f_matrix[4] = rolling_MEAN
f_matrix[5] = window_MEAN

# scale the derived matrix and test the model
test_features = the_scaler.fit_transform(f_matrix)
test_data_df = the_scaler.transform(test_features)
predictions_results = the_classifier.predict(test_data_df)

# save the results in a csv file
results = pd.DataFrame(data=predictions_results)
results.to_csv('Results.csv', header=None, index=None)
print("**********************END**********************")

#%% In [0]: Import libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.data import _create_method
# np.set_printoptions(threshold = np.inf)
# pd.options.display.max_columns = 20
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean

# In[1]: Look at the big picture

#%% In [2]: Get the data, load data
# Data is collected Kaggle(from www.kaggle.com/ionaskel/laptop-prices) and another source.
raw_data = pd.read_csv('datasets\laptops_kaggle_com_ionaskel.csv')

#%% In[3] Quick view of data
print('____________________________________ Dataset info ____________________________________')
print(raw_data.info())              
print('\n____________________________________ Some first data examples ____________________________________')
print(raw_data.head(6)) 
print('\n____________________________________ Counts on a feature ____________________________________')
print(raw_data['Cpu'].value_counts()) 
print('\n____________________________________ Statistics of numeric features ____________________________________')
print(raw_data.describe())    
print('\n____________________________________ Get specific rows and cols ____________________________________')     
print(raw_data.loc[[0,5,20], ['Company', 'Price_euros']] ) # Refer using column name
print(raw_data.iloc[[0,5,20], [2, 7]])

# In[4]: Scatter plot b/w 2 features
def plot_price_ram():
    raw_data.plot(kind='scatter', y="Price_euros", x="Ram", alpha=0.2)
    plt.savefig('figures/scatter_1_price_ram.png', format='png',dpi=400)
    plt.show()

print('_______________Scatter Price(euros) and Ram(GB) ______________')
plot_price_ram()

def plot_price_inches():
    raw_data.plot(kind='scatter', y="Price_euros", x="Inches", alpha=0.2)
    plt.savefig('figures/scatter_2_price_Inches.png', format='png',dpi=400)
    plt.show()

print('_______________Scatter Price(euros) and Inches ______________')
plot_price_inches()

# In[5]: Scatter plot b/w every pair of features
def plot_every_pair_of_features():
    from pandas.plotting import scatter_matrix
    features_to_plot = ["Price_euros","Inches","Ram","Weight"]
    scatter_matrix(raw_data[features_to_plot])
    plt.savefig('figures/scatter_every_pair_of_features.png', format='png', dpi=300)
    plt.show()
print('_______________Scatter every pair of features ______________')  
plot_every_pair_of_features()

# In[6]: Plot histogram of 1 feature
def plot_hist_of_1_feature():
    from pandas.plotting import scatter_matrix   
    features_to_plot = ["Ram"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8))
    plt.show()

plot_hist_of_1_feature()

# In[7]: Plot histogram of numeric features
def plot_hist_of_numberic_feature():
    raw_data.hist(figsize=(10,5)) #bins: no. of intervals
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.tight_layout()
    plt.savefig('figures/hist_raw_data(numberic).png', format='png', dpi=300) #
    plt.show()

plot_hist_of_numberic_feature()

# In[8]: Compute correlations b/w features
def compute_correlation_feature():
    corr_matrix = raw_data.corr()
    print(corr_matrix) # print correlation matrix
    print(corr_matrix["Price_euros"].sort_values(ascending=False)) # print correlation b/w a feature and other features

compute_correlation_feature()

# In[9]:Prepare the data: remove unused features
raw_data.drop(columns = ["No","TypeName","OpSys",""],inplace=True)

# In[10]: Prepare the data: Split training-test set and never touch test set until test phase
method = 2
if method == 1:
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42)
else:
    raw_data["aproxi_price"] = pd.cut(raw_data["Price_euros"],
                                    bins=[0, 300, 700, 1200, 2000, np.inf],
                                    #labels=["<2 tỷ", "2-4 tỷ", "4-6 tỷ", "6-8 tỷ", "8-10 tỷ", ">10 tỷ"])
                                    labels=[3,7,12,20,100]) # use numeric labels to plot histogram
    
    # Create training and test set
    from sklearn.model_selection import StratifiedShuffleSplit  
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # n_splits: no. of re-shuffling & splitting = no. of train-test sets 
                                                                                  # (if you want to run the algorithm n_splits times with different train-test set)
    for train_index, test_index in splitter.split(raw_data, raw_data["aproxi_price"]): # Feature "KHOẢNG GIÁ" must NOT contain NaN
        train_set = raw_data.loc[train_index]
        test_set = raw_data.loc[test_index]      
    
    # See if it worked as expected
    if 0:
        raw_data["aproxi_price"].hist(bins=6, figsize=(5,5)); #plt.show();
        test_set["aproxi_price"].hist(bins=6, figsize=(5,5)); plt.show()

    # Remove the new feature
    print(train_set.info())
    for _set_ in (train_set, test_set):
        #_set_.drop("income_cat", axis=1, inplace=True) # axis=1: drop cols, axis=0: drop rows
        _set_.drop(columns="aproxi_price", inplace=True) 
    print(train_set.info())
    print(test_set.info())

print('\n____________________________________ Split training an test set ____________________________________')     
print(len(train_set), "train +", len(test_set), "test examples")
print(train_set.head(4))

# In[11]: Prepare the data: Separate labels from data, since we do not process label values
train_set_labels = train_set["Price_euros"].copy()
train_set = train_set.drop(columns = "Price_euros") 
test_set_labels = test_set["Price_euros"].copy()
test_set = test_set.drop(columns = "Price_euros") 

# In[12]: Define pipeline for the processing data: a transformer for choosing columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values         

num_feature_names = ['Ram', 'Weight', 'Inches']
cat_feature_names = ['Product','ScreenResolution','Cpu','Memory','Gpu']
# In[13]: Pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feature_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors
    ]) 

# In[14]: Pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feature_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features to zero mean and unit variance
    ]) 

# In[15]: Combine features transformed by two above pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# In[16]: Run the pipeline to process training data 
processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________________________________ Processed feature values ____________________________________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)

#17 -> 20 : LinearRegression model
# In[17]: Train and evaluate models(LinearRegression model)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(processed_train_set_val, train_set_labels)
print('\n____________________________________ LinearRegression ____________________________________')
print('Learned parameters: ', model.coef_)

# In[18]: Compute R2 score and root mean squared error
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse      
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# In[19]: Predict labels for some training instances
print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))

# In[20]: Store models to files, to compare latter
import joblib # new lib
def store_model(model, model_name = ""):
    # NOTE: sklearn.joblib faster than pickle of Python
    # INFO: can store only ONE object in a file
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'saved_objects/' + model_name + '_model.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load('saved_objects/' + model_name + '_model.pkl')
    #print(model)
    return model
store_model(model)

# In[21]: Try DecisionTreeRegressor model
# Training
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________________________________ DecisionTreeRegressor ____________________________________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))

# In[22]: Try RandomForestRegressor model
# Training (NOTE: may take time if train_set is large)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5, random_state=42) # n_estimators: no. of trees
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________________________________ RandomForestRegressor ____________________________________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)      
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


# In[23]: Try polinomial regression model
# NOTE: polinomial regression can be treated as (multivariate) linear regression where high-degree features x1^2, x2^2, x1*x2... are seen as new features x3, x4, x5... 
# hence, to do polinomial regression, we add high-degree features to the data, then call linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_feat_adder = PolynomialFeatures(degree = 2) # add high-degree features to the data
train_set_poly_added = poly_feat_adder.fit_transform(processed_train_set_val)
new_training = 10
if new_training:
    model = LinearRegression()
    model.fit(train_set_poly_added, train_set_labels)
    store_model(model, model_name = "PolinomialRegression")      
else:
    model = load_model("PolinomialRegression")
# Compute R2 score and root mean squared error
print('\n____________________________________ Polinomial regression ____________________________________')
r2score, rmse = r2score_and_rmse(model, train_set_poly_added, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# Predict labels for some training instances
print("Predictions: ", model.predict(train_set_poly_added[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))
# In[24]: Evaluate with K-fold cross validation 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_predict
print('\n____________________________________ K-fold cross validation ____________________________________')
run_evaluation = 0
if run_evaluation:
    # Evaluate LinearRegression
    model_name = "LinearRegression" 
    model = LinearRegression()             
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))

    # Evaluate DecisionTreeRegressor
    model_name = "DecisionTreeRegressor" 
    model = DecisionTreeRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))

    # Evaluate RandomForestRegressor
    model_name = "RandomForestRegressor" 
    model = RandomForestRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    
    # Evaluate Polinomial regression
    model_name = "PolinomialRegression" 
    model = LinearRegression()
    nmse_scores = cross_val_score(model, train_set_poly_added, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
else:
    # Load rmse
    model_name = "LinearRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "DecisionTreeRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "RandomForestRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "PolinomialRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

# In[24]: Fine-tune models
# NOTE: this takes TIME
# INFO: find best hyperparams (param set before learning, e.g., degree of polynomial in poly reg, no. of trees in rand forest, no. of layers in neural net)
# Here we fine-tune RandomForestRegressor and PolinomialRegression
print('\n____________________________________ Fine-tune models ____________________________________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    print('Best estimator: ', grid_search.best_estimator_)  
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

# Grid search (try all combinations of hyperparams in param_grid)
def grid_search():
    from sklearn.model_selection import GridSearchCV
    
    run_new_search = 0     
    if run_new_search:
        # 6.1.1 Fine-tune RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]} ]
            # Train across 5 folds, hence a total of (12+6)*5=90 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")      

        # 6.1.2 Fine-tune Polinomial regression          
        model = Pipeline([ ('poly_feat_adder', PolynomialFeatures()), # add high-degree features
                           ('lin_reg', LinearRegression()) ]) 
        param_grid = [
            # try 3 values of degree
            {'poly_feat_adder__degree': [1, 2, 3]} ] # access param of a transformer: <transformer>__<parameter> https://scikit-learn.org/stable/modules/compose.html
            # Train across 5 folds, hence a total of 3*5=15 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/PolinomialRegression_gridsearch.pkl') 
        print_search_result(grid_search, model_name = "PolinomialRegression") 
    else:
        # Load grid_search
        grid_search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")         
        grid_search = joblib.load('saved_objects/PolinomialRegression_gridsearch.pkl')
        print_search_result(grid_search, model_name = "PolinomialRegression") 
grid_search()
# In[25] : Analyze and test the solution
search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
best_model = search.best_estimator_

# In[26]: 
print('\n____________________________________ ANALYZE AND TEST YOUR SOLUTION ____________________________________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUION")   

# In[27]: 
if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score  (ONLY on rand forest)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + onehot_cols
    for name in cat_feature_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

# In[28] : Run model on test data
# 7.3 Run on test data
test_set.drop('ZenBook 3','Thinkpad T560', 'PL60 7RD', 'V330-15IKB (i3-7130U/4GB/128GB/FHD/W10)', 'X555BP-XX180T (A9-9420/4GB/1TB/Radeon', 'V310-15ISK (i5-7200U/8GB/1TB')
test_set.drop('Chromebook Plus', 'Gram 14Z970', 'Chromebook 3', 'Inspiron 3179', 'IdeaPad 320-14IAP', 'Rog GL702VM-GC354T', 'Rog GL502VS', '17-ak001nv (A6-9220/4GB/500GB/Radeon', 'GL62 6QF')
processed_test_set = full_pipeline.transform(test_set)  
# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 7.3.2 Predict labels for some test instances
print("\nTest data: \n", test_set.iloc[0:9])
print("Predictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')

#In[29]: Maintain(Use Tkinter)

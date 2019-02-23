import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import graphviz
from sklearn import ensemble , preprocessing , metrics , svm , tree , linear_model
from sklearn.neural_network import MLPClassifier
### training data
datacust = pd.read_csv ('D:\\bigdata\\training_set\\train_cust_info.csv')
datatpye = pd.read_csv ('D:\\bigdata\\training_set\\train_tpy_info.csv')
databuy = pd.read_csv ('D:\\bigdata\\training_set\\train_buy_info.csv')
trainans = databuy.iloc [: , :2]
trainy = databuy ['BUY_TYPE']
datacustID = datacust.iloc [: , 0]
datatpyeID = datatpye.iloc [: , 0]

### test data
datacusttest = pd.read_csv ('D:\\bigdata\\training_set\\test_cust_x_info.csv')
datatpyetest = pd.read_csv ('D:\\bigdata\\training_set\\test_tpy_x_info.csv')
databuytest = pd.read_csv ('D:\\bigdata\\training_set\\test_buy_x_info.csv')
testID = databuytest.iloc [: , 0]
### fix na data

### testdatabuy
Wtest = databuytest ['WEIGHT'].mean (axis = 0)
Htest = databuytest ['HEIGHT'].mean (axis = 0)
testHW = {'HEIGHT' : Wtest , 'WEIGHT' : Htest}
databuytest = databuytest.fillna (value = testHW)
###testdatacust
datacusttest = datacusttest.dropna (axis = 1)
datacusttest = datacusttest.drop (['IS_EMAIL' , 'IS_PHONE' , 'IS_APP'] , axis = 1)
### traindatabuy
Wtrain = databuy ['WEIGHT'].mean (axis = 0)
Htrain = databuy ['HEIGHT'].mean (axis = 0)
trainHW = {'HEIGHT' : Wtrain , 'WEIGHT' : Htrain}
datatbuy = databuy.fillna (value = trainHW)
### traindatacust
datacust = datacust.dropna (axis = 1)

#### dum var use one-hot encoding

### Sex
testsex = databuytest ['SEX']
testsex = testsex == 'a'
databuytest ['SEX'] = testsex * 1
### traindata
trainsex = databuy ['SEX']
trainsex = trainsex == 'a'
databuy ['SEX'] = trainsex * 1

### Age
### testdata
testage = pd.get_dummies (databuytest ['AGE'])
testage.columns = ['AGE_a' , 'AGE_b' , 'AGE_c' , 'AGE_d' , 'AGE_e' , 'AGE_f' , 'AGE_g' , 'AGE_h' , 'AGE_i' , 'AGE_j' ,
                   'AGE_k' , 'AGE_l' , 'AGE_m' , 'AGE_n' , 'AGE_o' , 'AGE_p' , 'AGE_q']
###traindata
trainage = pd.get_dummies (databuy ['AGE'])
trainage.columns = ['AGE_a' , 'AGE_b' , 'AGE_c' , 'AGE_d' , 'AGE_e' , 'AGE_f' , 'AGE_g' , 'AGE_h' , 'AGE_i' , 'AGE_j' ,
                    'AGE_k' , 'AGE_l' , 'AGE_m' , 'AGE_n' , 'AGE_o' , 'AGE_p' , 'AGE_q']
###City code
###testdata
testcity = pd.get_dummies (databuytest ['CITY_CODE'])
testcity.columns = ['CITY_a' , 'CITY_b' , 'CITY_c' , 'CITY_d' , 'CITY_e' , 'CITY_f' , 'CITY_g' , 'CITY_h' , 'CITY_i' ,
                    'CITY_j' , 'CITY_k' , 'CITY_l' , 'CITY_m' , 'CITY_n' , 'CITY_o' , 'CITY_p' , 'CITY_q' , 'CITY_r' ,
                    'CITY_s' , 'CITY_t' , 'CITY_u' , 'CITY_v' , 'CITY_w']
###traindata
traincity = pd.get_dummies (databuy ['CITY_CODE'])
traincity.columns = ['CITY_a' , 'CITY_b' , 'CITY_c' , 'CITY_d' , 'CITY_e' , 'CITY_f' , 'CITY_g' , 'CITY_h' , 'CITY_i' ,
                     'CITY_j' , 'CITY_k' , 'CITY_l' , 'CITY_m' , 'CITY_n' , 'CITY_o' , 'CITY_p' , 'CITY_q' , 'CITY_r' ,
                     'CITY_s' , 'CITY_t' , 'CITY_u' , 'CITY_v' , 'CITY_w']
###OCCPUATION
###testdata
testocc = pd.get_dummies (databuytest ['OCCUPATION'])
###traindata
trainocc = pd.get_dummies (databuy ['OCCUPATION'])
### find the same occ
finsamocc = pd.concat ([trainocc , testocc] , sort = True , axis = 0)
occsam = finsamocc.dropna (axis = 1)
occsamlist = occsam.columns
trainocc = trainocc [occsamlist]

###Marriage
###testdata
testmar = pd.get_dummies (databuytest ['MARRIAGE'])
testmar.columns = ['MARRIAGE_a' , 'MARRIAGE_b' , 'MARRIAGE_c' , 'MARRIAGE_d' , 'MARRIAGE_e' , 'MARRIAGE_f']

###traindata
trainmar = pd.get_dummies (databuy ['MARRIAGE'])
trainmar.columns = ['MARRIAGE_a' , 'MARRIAGE_b' , 'MARRIAGE_c' , 'MARRIAGE_d' , 'MARRIAGE_e' , 'MARRIAGE_f']

###Edu
###testdata
testedu = pd.get_dummies (datacusttest ['EDUCATION'])
testedu.columns = ['edu_a' , 'edu_b' , 'edu_c' , 'edu_d']
###traindata
trainedu = pd.get_dummies (datacust ['EDUCATION'])
trainedu.columns = ['edu_a' , 'edu_b' , 'edu_c' , 'edu_d']
###par dead
###testdata
testparns = pd.get_dummies (datacusttest ['PARENTS_DEAD'])
testparns.columns = ['parns_a' , 'parns_b']
###traindata
trainparns = pd.get_dummies (datacust ['PARENTS_DEAD'])
trainparns.columns = ['parns_a' , 'parns_b']
### real estate
###testdata
testest = pd.get_dummies (datacusttest ['REAL_ESTATE_HAVE'])
testest.columns = ['est_a' , 'eat_b']
###traindata
trainest = pd.get_dummies (datacust ['REAL_ESTATE_HAVE'])
trainest.columns = ['est_a' , 'eat_b']
### major income
###testdata
testmaj = pd.get_dummies (datacusttest ['IS_MAJOR_INCOME'])
testmaj.columns = ['maj_a' , 'maj_b']
###traindata
trainmaj = pd.get_dummies (datacust ['IS_MAJOR_INCOME'])
trainmaj.columns = ['maj_a' , 'maj_b']
###buying tpye
###testdata
testtpy1 = pd.get_dummies (datatpyetest ['BUY_TPY1_NUM_CLASS'])
testtpy1.columns = ['TPY1_A' , 'TPY1_B' , 'TPY1_C' , 'TPY1_D' , 'TPY1_E' , 'TPY1_F' , 'TPY1_G']
testtpy2 = pd.get_dummies (datatpyetest ['BUY_TPY2_NUM_CLASS'])
testtpy2.columns = ['TPY2_A' , 'TPY2_B' , 'TPY2_C' , 'TPY2_D' , 'TPY2_E' , 'TPY2_F' , 'TPY2_G']
testtpy3 = pd.get_dummies (datatpyetest ['BUY_TPY3_NUM_CLASS'])
testtpy3.columns = ['TPY3_D' , 'TPY3_E' , 'TPY3_F' , 'TPY3_G']
testtpy4 = pd.get_dummies (datatpyetest ['BUY_TPY4_NUM_CLASS'])
testtpy4.columns = ['TPY4_B' , 'TPY4_C' , 'TPY4_D' , 'TPY4_E' , 'TPY4_F' , 'TPY4_G']
testtpy5 = pd.get_dummies (datatpyetest ['BUY_TPY5_NUM_CLASS'])
testtpy5.columns = ['TPY5_C' , 'TPY5_D' , 'TPY5_E' , 'TPY5_F' , 'TPY5_G']
testtpy6 = pd.get_dummies (datatpyetest ['BUY_TPY6_NUM_CLASS'])
testtpy6.columns = ['TPY6_B' , 'TPY6_C' , 'TPY6_D' , 'TPY6_E' , 'TPY6_F' , 'TPY6_G']
testtpy7 = pd.get_dummies (datatpyetest ['BUY_TPY7_NUM_CLASS'])
testtpy7.columns = ['TPY7_A' , 'TPY7_B' , 'TPY7_C' , 'TPY7_D' , 'TPY7_E' , 'TPY7_F' , 'TPY7_G']
###traindata
traintpy1 = pd.get_dummies (datatpye ['BUY_TPY1_NUM_CLASS'])
traintpy1.columns = ['TPY1_A' , 'TPY1_B' , 'TPY1_C' , 'TPY1_D' , 'TPY1_E' , 'TPY1_F' , 'TPY1_G']
traintpy2 = pd.get_dummies (datatpye ['BUY_TPY2_NUM_CLASS'])
traintpy2.columns = ['TPY2_A' , 'TPY2_B' , 'TPY2_C' , 'TPY2_D' , 'TPY2_E' , 'TPY2_F' , 'TPY2_G']
traintpy3 = pd.get_dummies (datatpye ['BUY_TPY3_NUM_CLASS'])
traintpy3 = traintpy3.drop ('C' , axis = 1)
traintpy3.columns = ['TPY3_D' , 'TPY3_E' , 'TPY3_F' , 'TPY3_G']
traintpy4 = pd.get_dummies (datatpye ['BUY_TPY4_NUM_CLASS'])
traintpy4 = traintpy4.drop ('A' , axis = 1)
traintpy4.columns = ['TPY4_B' , 'TPY4_C' , 'TPY4_D' , 'TPY4_E' , 'TPY4_F' , 'TPY4_G']
traintpy5 = pd.get_dummies (datatpye ['BUY_TPY5_NUM_CLASS'])
traintpy5 = traintpy5.drop (['A' , 'B'] , axis = 1)
traintpy5.columns = ['TPY5_C' , 'TPY5_D' , 'TPY5_E' , 'TPY5_F' , 'TPY5_G']
traintpy6 = pd.get_dummies (datatpye ['BUY_TPY6_NUM_CLASS'])
traintpy6 = traintpy6.drop ('A' , axis = 1)
traintpy6.columns = ['TPY6_B' , 'TPY6_C' , 'TPY6_D' , 'TPY6_E' , 'TPY6_F' , 'TPY6_G']
traintpy7 = pd.get_dummies (datatpye ['BUY_TPY7_NUM_CLASS'])
traintpy7.columns = ['TPY7_A' , 'TPY7_B' , 'TPY7_C' , 'TPY7_D' , 'TPY7_E' , 'TPY7_F' , 'TPY7_G']
###concat all data together

###testdata
###buy
testbuy = pd.concat ([databuytest , testage , testcity , testocc , testmar] , axis = 1)
testbuy = testbuy.drop (['AGE' , 'BUY_YEAR' , 'MARRIAGE' , 'CITY_CODE' , 'OCCUPATION'] , axis = 1)
###cust
testcust = pd.concat ([datacusttest , testedu , testparns , testest , testmaj] , axis = 1)
testcust = testcust.drop (['EDUCATION' , 'PARENTS_DEAD' , 'REAL_ESTATE_HAVE' , 'IS_MAJOR_INCOME'] ,
                          axis = 1)
###tpye
testtpye = pd.concat ([datatpyetest , testtpy1 , testtpy2 , testtpy3 , testtpy4 , testtpy5 , testtpy6 , testtpy7] ,
                      axis = 1)
testtpye = testtpye.drop (['BUY_TPY1_NUM_CLASS' , 'BUY_TPY2_NUM_CLASS' , 'BUY_TPY3_NUM_CLASS' , 'BUY_TPY4_NUM_CLASS' ,
                           'BUY_TPY5_NUM_CLASS' , 'BUY_TPY6_NUM_CLASS' , 'BUY_TPY7_NUM_CLASS'] , axis = 1)

###traindata
###buy
trainbuy = pd.concat ([databuy , trainage , traincity , trainocc , trainmar] , axis = 1)
trainbuy = trainbuy.drop (['BUY_TYPE' , 'AGE' , 'BUY_YEAR' , 'MARRIAGE' , 'CITY_CODE' , 'OCCUPATION'] , axis = 1)
###cust
traincust = pd.concat ([datacust , trainedu , trainparns , trainest , trainmaj] , axis = 1)
traincust = traincust.drop (['EDUCATION' , 'PARENTS_DEAD' , 'REAL_ESTATE_HAVE' , 'IS_MAJOR_INCOME'] ,
                            axis = 1)
###tpye
traintpye = pd.concat ([datatpye , traintpy1 , traintpy2 , traintpy3 , traintpy4 , traintpy5 , traintpy6 , traintpy7] ,
                       axis = 1)
traintpye = traintpye.drop (['BUY_TPY1_NUM_CLASS' , 'BUY_TPY2_NUM_CLASS' , 'BUY_TPY3_NUM_CLASS' , 'BUY_TPY4_NUM_CLASS' ,
                             'BUY_TPY5_NUM_CLASS' , 'BUY_TPY6_NUM_CLASS' , 'BUY_TPY7_NUM_CLASS'] , axis = 1)

########### mix all data use cust_id
###test data
testdataall = pd.merge (testbuy , testcust , on = 'CUST_ID')
testdataall = pd.merge (testdataall , testtpye , on = 'CUST_ID')
###train data
traindata = pd.merge (trainbuy , traincust , on = 'CUST_ID')
traindata = pd.merge (traindata , traintpye , on = 'CUST_ID')

###drop cust_id
testdataall = testdataall.drop (['CUST_ID'] , axis = 1)
traindata = traindata.drop (['CUST_ID'] , axis = 1)

###refillna
traindata ['HEIGHT'] = traindata ['HEIGHT'].fillna (Htrain)
traindata ['WEIGHT'] = traindata ['WEIGHT'].fillna (Wtrain)
##########build model
train_X , test_X , train_y , test_y = train_test_split (traindata , trainy , test_size = 0.3)
####tree
treeclf = tree.DecisionTreeClassifier ()
tree_clf = treeclf.fit (train_X , train_y)
print ('the score of tree = ' , tree_clf.score (test_X , test_y))
#####random forest
forest = ensemble.RandomForestClassifier (n_estimators = 600)
forest_fit = forest.fit (train_X , train_y)
print ('the score of forest = ' , forest_fit.score (test_X , test_y))
### logistic
logistic_regr = linear_model.LogisticRegression ()
logistic_regr.fit (train_X , train_y)
print ('the score of logistic = ' , logistic_regr.score (test_X , test_y))
#### svm
svc = svm.SVC ()
svc_fit = svc.fit (train_X , train_y)
print ('the score of svm = ' , svc_fit.score (test_X , test_y))

#####picture of tree
class_names = ['A' , 'B' , 'C' , 'D' , 'E' , 'F' , 'G']
featurenames = traindata.columns.tolist ()
dot_data = tree.export_graphviz (treeclf , out_file = None , feature_names = featurenames ,
                                 class_names = class_names , max_depth = 5 ,
                                 filled = True , rounded = True ,
                                 special_characters = True)
graph = graphviz.Source (dot_data)
graph.render ('finaltree')

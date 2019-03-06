import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt # 可视化模块
from sklearn.model_selection import train_test_split
# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()

X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2)# test 后 40 data points

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='adam')

# training
print('Training -----------')
print('Training -----------')
cost = model.fit(X_train,Y_train,batch_size= 50 ,epochs = 100)

        
        
# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=10)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)


# plotting the prediction
Y_pred = model.predict(X_train)
plt.scatter(X_test, Y_test)
plt.plot(X_train, Y_pred)
plt.show()


#Define the model
def baseline_model():
   model = Sequential()
   model.add(Dense(1, activation = 'linear', input_dim = 1))
   model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
   return model
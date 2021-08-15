import tensorflow as tf
import os
import pickle
import numpy as np
import pandas as pd


from tensorflow import keras
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras import Model

from tensorflow.keras import Sequential


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.stats import zscore


#download data
path = 'data_preprocessed_python'
path_list = os.listdir(path)

dataset_x, dataset_y = [], []

path_list.sort()

for filename in path_list:
    doc = open(os.path.join(path, filename), 'rb')
    data = pickle.load(doc, encoding = 'bytes')
    label = data[list(data.keys())[0]]
    signal = data[list(data.keys())[1]]
    #print(label.shape,signal.shape)
    #for i in range(40):
    dataset_x.append(signal)
    dataset_y.append(label)

#len(dataset_x), dataset_x[0].shape


del label; del signal;
len(dataset_x), dataset_x[0].shape   #person,music,channel


x = np.asarray(dataset_x)
y = np.asarray(dataset_y)
del dataset_x; del dataset_y;


x = x[:,:,:,:,np.newaxis]
x = zscore(x, axis = 1)  
y = y/9


np.set_printoptions(threshold=np.inf)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)


x_train = x_train.transpose(2,0,1,3,4)
x_test = x_test.transpose(2,0,1,3,4)
x_valid = x_valid.transpose(2,0,1,3,4)
x_train.shape,x_valid.shape,x_test.shape


METRICS = [
      keras.metrics.MeanAbsoluteError(name='MAE'),
      keras.metrics.MeanAbsolutePercentageError(name='MAPE'),
      keras.metrics.MeanSquaredError(name='MSE'),
      keras.metrics.MeanSquaredLogarithmicError(name='MSLE'),
]

def create_models(dense_par=20, sub_signals=12, metrics = METRICS, model_dim1=25):
    
    #sample_size = int(8064/sub_signals)

    models = [0]*model_dim1
    
    for i in range(model_dim1):
        models[i] = Sequential()
        #block 1
        models[i].add(Conv1D(filters=32, kernel_size=11,strides = 3))
        models[i].add(BatchNormalization())
        models[i].add(tf.keras.layers.Activation('relu'))

        #block 2
        models[i].add(Conv1D(filters=24, kernel_size=3,strides = 2))
        models[i].add(BatchNormalization())
        models[i].add(tf.keras.layers.Activation('relu'))

        #block 3
        models[i].add(Conv1D(filters=16, kernel_size=3,strides = 2))
        models[i].add(BatchNormalization())
        models[i].add(tf.keras.layers.Activation('relu'))

        #block 4
        models[i].add(Conv1D(filters=8, kernel_size=3,strides = 2))
        models[i].add(BatchNormalization())
        models[i].add(tf.keras.layers.Activation('relu'))

        #fc-1
        models[i].add(Flatten())
        models[i].add(Dense(dense_par, activation='relu'))

        #dropout
        models[i].add(Dropout(rate = 0.5))

        models[i].add(Dense(4))
        #models[i].compile(optimizer= tf.keras.optimizers.Adam(learning_rate=1e-4) , metrics= metrics)
        models[i].compile(optimizer= tf.keras.optimizers.Adam(learning_rate=1e-4) , 
                          loss = tf.keras.losses.mean_squared_error, metrics= metrics)
        #MAE:mean_absolute_error
    
    print("All models defined.")
    return models


models = create_models(dense_par = 20, sub_signals = 12, 
                       metrics = METRICS, model_dim1=x_train.shape[2])

history = [0] * x_train.shape[2]


#x_train.shape[2]
for j in range(x_train.shape[2]):
    print("==============================Music/Video {:02d}==============================".format(j+1))
    
    history[j] = models[j].fit(x_train[0,:,j], y_train[:,j], epochs = 200, 
              validation_data = (x_valid[0,:,j], y_valid[:,j]), shuffle = True)
    models[j].save("model_dir/model_"+str(j)+'.h5')

#plt.plot(history[0].history['loss'])

#plt.savefig(results_dir + '/acc.png',  bbox_extra_artists=(lgd,), bbox_inches='tight')


plt.style.use('ggplot')
for k in range(len(history)):
    print('total_musics=',len(history))
    plt.figure()
    plt.plot(history[k].history['MAE'],label='MAE')
    plt.plot(history[k].history['MSE'],label='MSE')
    #plt.plot(history[0].history['MAPE'],label='MAPE')
    plt.legend()
    plt.savefig('result_dir/'+'music_'+str(k)+'train'+'.png')
    plt.show()


predict = [0] * x_train.shape[2]
for j in range(x_train.shape[2]):
    predict[j] = models[j].evaluate(x_test[0,:,j], y_test[:,j])

result_array = np.zeros((40,5))
for i in range(len(predict)):
    for j in range(len(predict[0])):
        result_array[i][j] = predict[i][j]
print(result_array)



test_result = pd.DataFrame(result_array,
	columns= ['loss','MAE', 'MAPE','MSE','MSLE'],
	index=None)

print(test_result)

#plt.plot(test_result['loss'],label='ordinary_loss')
plt.figure()
plt.plot(test_result['MAE'],label='MAE')
plt.plot(test_result['MSE'],label='MSE')

plt.xlabel('music')
plt.ylabel('loss')
plt.legend()
plt.savefig('result_dir/'+'test_result'+'.png')
plt.show()
test_result.to_csv('result_dir/test_result.csv')






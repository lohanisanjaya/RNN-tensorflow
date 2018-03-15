'''Correspondence:
   SL (slohani@tulane.edu)
   RTG (rglasser@tulane.edu)
   Feb 17, 2018'''

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import cPickle as pkl
#import _pickle as pkl
import time

class Local_Rnn_LSTM(object):

        def __init__(self, eta = 0.001, epochs = 20, batch_size = 5, time_steps = 6,\
                          num_inputs = 5, hidden_units = 20, num_classes = 10):

            #Training Parameters
            self.eta = eta
            self.epochs = epochs	#Generally Iterations = epochs*batch_size
            self.batch_size = batch_size 
            self.nor = 0.
            #Network Parametes
            #data are fed to LSTM row-wise
            self.time_steps = time_steps #number of rows data
            self.num_inputs = num_inputs #number of columns data
            self.hidden_units = hidden_units #number of hidden neurons in ANN, dimension of weight
            self.num_classes = num_classes   #number of output neurons

        def Pkl_Read(self,filename):
            file_open = open(filename,'rb')
            value = pkl.load(file_open)
            return value

        def norm(self,data,minm=0,maxm=255):
            data = (data - np.min(data))/(np.max(data)-np.min(data))
            data = data*(maxm-minm)+minm
            return data 

        def Pkl_Save(self,filename,value):
            file_open = open(filename,'wb')
            return pkl.dump(value,filename,protocol=pkl.HIGHEST_PROTOCOL)

        def Data_Loader(self,data_file = 'data.pkl'):
            ''' It creats an iterator of training sets with respect to batch_size. We can call
                this using next(iterator) command while running the optimization.
                The data should be list of [[array_X.flatten],[labels_Y]]. For eg, if you want to have 10
                training images (128x128 pixs) with labels ranging from 0 to 9 then 
                                   data = X,Y = [[array_flatten[0, .. .. ..]_128x128,
                                           array[0., 0.. .. ..]_128x128,
                                           array[0, .. .. ..]_128x128,
                                            .. ... .... ... ... ..], array[0,1,2,3,4,5,6,7,8,9]]
                For debuging read at https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/ '''
            data_X,data_Y = self.Pkl_Read(filename = data_file)
            self.nor = np.linalg.norm(data_X)
            data_X = data_X/self.nor
            self.data_X = data_X
            print ('norm', self.nor)
            #assert len(data_X[0]) == (self.time_steps*self.num_inputs), 'Time steps x Num_inputs \
            #                                           does not match with dimension of input data'
            #*#For time-series
            #assert len(data_X) == (self.time_steps*self.num_inputs), 'Time steps x Num_inputs \
            #                                           does not match with dimension of input data'
            #data_X = data_X.reshape(self.batch_size,self.time_steps,self.num_inputs)
            data_X_seq = [np.array(data_X[i * self.num_inputs: (i + 1) * self.num_inputs]) for i\
                       in range(len(data_X) // self.num_inputs)]
            data_X_out = np.array([data_X_seq[i: i + self.time_steps] for i in range(len(data_X_seq\
                        )- self.time_steps)])
            data_Y_out = np.array([data_X_seq[i + self.time_steps] for i in range(len(data_X_seq)\
                          - self.time_steps)]) #[0] for only one output prediction/ next element in seq
            print ('X {}, shape {}'.format(data_X_out*self.nor ,data_X_out.shape))
            print ('Y',data_Y_out*self.nor)

            #------------------------------------------------------------
            #*#For one_hot type outputs <----------------
            ##shuffle the data
            #zipped = list(zip(data_X,data_Y))
            #np.random.shuffle(zipped)
            #data_X,data_Y = list(zip(*zipped))
            ###X_batch = np.array(data_X).reshape(self.batch_size,self.time_steps,self.num_inputs)
            ###generating one hot form for Y_batch
            #indices = np.array(data_Y)
            #depth = self.num_classes
            #data_Y = tf.one_hot(indices,depth) #Dimension of len(data_Y) X num_classes 
            #----------------------------------------------------------------------------------
            return [data_X,data_X_out,data_Y_out] #data_X is a list of [number of training sets X (time_steps*num_inputs)]
        
        def LSTM(self):
            return rnn.BasicLSTMCell(self.hidden_units,forget_bias=1.0)
        def Build_BasicLSTM(self,X_data,weights,biases):
            #current data input shape is (batch_size,time_steps,num_inputs)
            X = tf.unstack(X_data,self.time_steps,axis=1)
            #X = X_data
            #LSTM_CELL = rnn.BasicLSTMCell(self.hidden_units,forget_bias=1.0)
            #LSTM_CELL = rnn.LSTMCell(self.hidden_units)
            Multi_LSTM_CELL = rnn.MultiRNNCell([self.LSTM() for _ in range(1)],state_is_tuple=True)
            outs,states = tf.nn.static_rnn(Multi_LSTM_CELL,X,dtype=tf.float32)
            #print (sess.run(outs[-1].shape))          
            return outs[-2],tf.matmul(outs[-1],weights['final'])+biases['final']

        def Run_LSTM(self,data_file = 'data.pkl'):
            start = time.time()
            X = tf.placeholder(tf.float32,[None,self.time_steps,self.num_inputs])
            Y = tf.placeholder(tf.float32,[None,self.num_classes])
            weights = {'final': tf.Variable(tf.random_normal([self.hidden_units,self.num_classes]))}
            biases = {'final': tf.Variable(tf.random_normal([self.num_classes]))}
            out,logits = self.Build_BasicLSTM(X,weights,biases)
            #----------------------------------------------------------------------       
            #*#For time-series inputs #Y has [1] dimension <--------
            Loss = tf.reduce_mean(tf.square(logits - Y))
            Optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.eta)
            #Optimizer = tf.train.AdagradOptimizer(learning_rate=self.eta)
            Training = Optimizer.minimize(Loss)
            prediction_series = logits
            #-----------------------------------------------------------------------
            #*#For softmax type output/one_hot kind output <---
            #------------------------------------------------------------------------------------
            #final_predictions = tf.nn.softmax(logits) 
            ##LOSS and OPTIMIZER
            #Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))
            #Optimizer = tf.train.AdagradOptimizer(learning_rate=self.eta)
            #Training = Optimizer.minimize(Loss)
            # Evaluate model (with test logits, for dropout to be disabled)
            #correct_pred = tf.equal(tf.argmax(final_predictions, 1), tf.argmax(Y, 1))
            #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            #--------------------------------------------------------------------------------------
            # Initialize the variables (i.e. assign their default value)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            #Load the data
            true_X,X_data,Y_data = self.Data_Loader(data_file = data_file)     
            #For only one_hot type data <-------           
            #Y_data = sess.run(Y_data)  #feed data should not be in tensor format
            #plot_g = []
            for num_epoch in range(self.epochs):
                print ('')
                print ('|Epoch: {}'.format(num_epoch+1))
                print ('')
                gg = []
                for i in range(int(len(X_data)/self.batch_size)):
                  for k in range(1):
                    #print (i)
                    #gg = []
                    #for j in range(30):
                #    i = 0
                    #gg = []
                    X_batch = np.array(X_data[i*self.batch_size:(i+1)*self.batch_size]).reshape(\
                              self.batch_size,self.time_steps,self.num_inputs)
                    Y_batch = Y_data[i*self.batch_size:(i+1)*self.batch_size].reshape(\
                              self.batch_size,self.num_classes)
                    
                    sess.run(Training,feed_dict={X:X_batch,Y:Y_batch})
                    log,w,outp = sess.run([logits,weights,out],feed_dict={X:X_batch,Y:Y_batch})
                    #print ('logits',log.shape)
                    #print ('wt',w.get('final').shape)
                    #print ('out',outp.shape)
                    #accu,loss = sess.run([accuracy,Loss],feed_dict={X:X_batch,Y:Y_batch})
                    #print ('|Batch: {}  Accuracy: {} Loss: {}'.format(i+1,accu,loss))#<----
                    ##for series-data
                  pred,loss = sess.run([prediction_series,Loss],feed_dict={X:X_batch,Y:Y_batch})
                  gg.append(np.array(pred)*self.nor)
                    #print ('pred :{}'.format(np.array(pred)*self.nor))
            #Test set
            #test_X = self.Pkl_Read('data_2.pkl')
            #test_X = test_X/self.nor
            #test_X = test_X.reshape(self.batch_size,self.time_steps,self.num_inputs)
            #test_X = np.array(self.data_X[300:330]).reshape(1,10,3)
            test_X = np.array(self.data_X[270:330]).reshape(1,10,6)
            predicted_values = []
            predicted_binary = []
            for k in range(5):
                pred = sess.run([prediction_series],feed_dict={X:test_X})
                pred_test_binary = np.array(pred)
                print (pred_test_binary)
                predicted_binary.append(pred_test_binary)
                pred_appended = np.array(predicted_binary).flatten()
                new_test = np.concatenate((np.array(self.data_X[276+6*k:330]),pred_appended),axis=0)
              #  new_test = np.concatenate((np.array(self.data_X[60+3*k:330]),pred_appended),axis=0)
                test_X = new_test.reshape(1,10,6)
                predicted_values.append(np.array(pred)*self.nor)
            plot_g = np.concatenate((np.array(gg).flatten(),np.array(predicted_values).flatten()),\
                     axis=0)
            print ('plot_g',list(plot_g))   
            
            plt.plot(range(60,360),np.array(plot_g))
            #plt.plot(range(300,360),np.array(plot_g))
            plt.plot(range(330),true_X*self.nor)
            #print (np.array(pred)*self.nor)     
            plt.show()
            end = time.time()
            print ('')
            print ('-'*15+'{} s'.format(end-start)+'-'*15)
            print ('*'*15+'Happy Computing'+'*'*15)
            print ('*'*17+'Quinlo Group'+'*'*17)


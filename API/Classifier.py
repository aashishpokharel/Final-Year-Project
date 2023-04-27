import numpy as np
import pandas as pd

class Classifier:
    def __init__(self, n_inputs, n_neurons = [32,32,10], initialization = 'random', seed = 42):
        np.random.seed(seed)
        # We have done here n_inputs/n_neurons instead of n_neurons/n_inputs to prevent the Transpose everytime
        if initialization == 'random':
            self.weights1   = 0.01 * np.random.randn(n_inputs, n_neurons[0]) # The input shape and no of neurons you want to have in the layer
            self.biases1    =  0.01 * np.random.randn(1, n_neurons[0])
            self.weights2   = 0.01 *np.random.randn(n_neurons[0], n_neurons[1]) # The input shape and no of neurons you want to have in the layer
            self.biases2    = 0.01 * np.random.randn(1, n_neurons[1])
            self.weights3   = 0.01 * np.random.randn(n_neurons[1], n_neurons[2])
            self.biases3    = 0.01 * np.random.randn(1, n_neurons[2])
        if initialization == 'He':
            self.weights1   =  np.sqrt(2/n_neurons[0]) * np.random.randn(n_inputs, n_neurons[0]) # The input shape and no of neurons you want to have in the layer
            self.biases1    =  np.sqrt(2/n_neurons[0]) * np.random.randn(1, n_neurons[0])
            self.weights2   =  np.sqrt(2/n_neurons[1]) *np.random.randn(n_neurons[0], n_neurons[1]) # The input shape and no of neurons you want to have in the layer
            self.biases2    =  np.sqrt(2/n_neurons[1]) * np.random.randn(1, n_neurons[1])
            self.weights3   =  np.sqrt(2/n_neurons[2]) * np.random.randn(n_neurons[1], n_neurons[2])
            self.biases3    =  np.sqrt(2/n_neurons[2]) * np.random.randn(1, n_neurons[2])
        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.X = None
        self.y = None
        # activatied outputs
        self.output1_act = None
        self.output2_act = None
        self.output3_act = None
    def forward(self, inputs, weights, biases):
        """ The dot product of the input - weights - Biases (y = Wx + b) """
        output = np.dot(inputs, weights) + biases
        if(np.isnan(np.sum(output))):
            raise Exception("NaN values present in FW pass")
        elif(np.isinf(np.sum(output))):
            raise Exception("INF values present in FW Pass")
        
        return output

    def ReLU(self, inputs):
        """ Rectified Linear Activation Function """
        output = np.maximum(0, inputs)
        return output
    
    def Softmax(self, inputs):
    # subtract largest value to prevent overflow
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        if(np.isnan(np.sum(probabilities))):
            raise Exception("NaN values present in Softmax For")
        elif(np.isinf(np.sum(probabilities))):
            raise Exception("INF values present in Softmax For")
        
        return probabilities
        
    def categorical_cross_entropy(self,y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-6, 1-1e-6)
        # Handling if labels are 1D 
        correct_confidences = None
        if len(y_true.shape) == 1:
#             print(y_pred_clipped[range(samples), :].shape)
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) ==2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis =1)
        else:
            raise Exception("Sorry, no numbers below zero")
        
        negative_log_likelihoods = -np.log(correct_confidences)
#         print(negative_log_likelihoods.shape)
        return negative_log_likelihoods 
    
    def linear_backward(self,inputs, weights, dvalues):
        self.dweights_linear = np.dot(inputs.T, dvalues)
        self.dbiases_linear = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinput_linear = np.dot(dvalues, weights.T)
        
        if(np.isnan(np.sum(self.dweights_linear))):
            raise Exception("NaN values present in Linear Back")
        elif(np.isinf(np.sum(self.dweights_linear))):
            raise Exception("INF values present in Linear BAck")
        
        
        return self.dweights_linear, self.dinput_linear
    
    def linear_backward_with_l2(self,inputs, weights, dvalues, lambd = 0.5):
        """  """
        m = inputs.shape[1]
        self.dweights_linear = np.dot(inputs.T, dvalues) + (lambd*weights)/m
        self.dbiases_linear = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinput_linear = np.dot(dvalues, weights.T) 
        
        if(np.isnan(np.sum(self.dweights_linear))):
            raise Exception("NaN values present in Linear Back")
        elif(np.isinf(np.sum(self.dweights_linear))):
            raise Exception("INF values present in Linear BAck")
        
        
        return self.dweights_linear, self.dinput_linear
    
    def softmax_backward(self,dA, Z):
        """Compute backward pass for softmax activation"""
        softmax_output = self.Softmax(Z) 
        return softmax_output * (1 - softmax_output) * dA

    def ReLU_backward(self,dA, Z):
        
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        if(np.isnan(np.sum(dZ))):
            raise Exception("NaN values present in RELU Back")
        elif(np.isinf(np.sum(dZ))):
            raise Exception("INF values present in RELU BAck")
        return dZ
        
    def categorical_cross_entropy_backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs_loss = self.dinputs / samples
        if(np.isnan(np.sum(self.dinputs))):
            raise Exception("NaN values present in Softmax Back")
        elif(np.isinf(np.sum(self.dinputs))):
            raise Exception("INF values present in Softmax_back")
        return self.dinputs
    
    def softmax_categorical_cross_entropy_combined_backward(self, dvalues, y_true):
        samples = len(dvalues)
        #handling Ohe values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs_combined = dvalues.copy()
        # Calculate gradient
        self.dinputs_combined[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs_combined = self.dinputs_combined / samples
        if(np.isnan(np.sum(self.dinputs_combined))):
            raise Exception("NaN values present in Softmax Back")
        elif(np.isinf(np.sum(self.dinputs_combined))):
            raise Exception("INF values present in Softmax_back")
       
        return self.dinputs_combined
        
    
    def compute_loss(self,y_pred, y_true):
        sample_losses = self.categorical_cross_entropy(y_pred, y_true)
        loss = np.mean(sample_losses)
        return loss
    
    def compute_loss_with_l2(self,y_pred, y_true, lambd = 0.5):
        m = 10
        sample_losses = self.categorical_cross_entropy(y_pred, y_true)
        L2_regularization_cost = (lambd/(2*m))*(np.sum(np.square(self.weights1) + np.sum(np.square(self.weights2) + np.sum(np.square(self.weights3)))))
        loss = np.mean(sample_losses) 
        return loss
    
    
    def forward_pass(self, X):
        self.X = X
        self.output1     = self.forward(self.X, self.weights1, self.biases1)
        self.output1_act = self.ReLU(self.output1)
        self.output2     = self.forward(self.output1_act, self.weights2, self.biases2)
        self.output2_act = self.ReLU(self.output2)
        self.output3     = self.forward(self.output2_act, self.weights3, self.biases3)
        self.output3_act = self.Softmax(self.output3)
#         print("Softmax SUM", np.sum(self.output3_act, axis = 1))
        if(np.isnan(np.sum(self.output3_act))):
            raise Exception("NaN values present in data")
        elif(np.isinf(np.sum(self.output3_act))):
            raise Exception("INF values present in data")
        
        
    def check_inf(self):
        check_weights = np.any(np.isinf(self.weights1)) or np.any(np.isinf(self.weights2)) or np.any(np.isinf(self.weights3))
        check_bias    = np.any(np.isinf(self.biases1)) or np.any(np.isinf(self.biases2)) or np.any(np.isinf(self.biases3))
        return (check_weights or check_bias)
    
    def bgd(self, X, y, learning_rate= 0.1, iteration = 10000):
        self.X = X
        self.y = y
        for i in range(iteration):
            self.forward_pass(self.X)
            predictions = np.argmax(self.output3_act, axis=1)
            
            
            gradient_output3_act                  = self.softmax_categorical_cross_entropy_combined_backward(self.output3_act, self.y)
            gradient_output3, gradient_input3     = self.linear_backward(self.output2,self.weights3,gradient_output3_act)
            gradient_output2_act                  = self.ReLU_backward(gradient_input3, self.output2)
            gradient_output2, gradient_input2     = self.linear_backward(self.output1, self.weights2, gradient_output2_act)
            gradient_output1_act                  = self.ReLU_backward(gradient_input2, self.output1)
            gradient_output1, gradient_input1     = self.linear_backward(self.X, self.weights1, gradient_output1_act)
            
            self.weights3  = self.weights3 - learning_rate * gradient_output3
            self.weights2  = self.weights2 - learning_rate * gradient_output2
            self.weights1  = self.weights1 - learning_rate * gradient_output1
            assert np.sum(gradient_output1) != np.nan, "The gradient has nan"
            assert np.sum(gradient_output1) != np.inf, "The gradient has inf"
            if i%100 == 0:

                loss = self.compute_loss(self.output3_act, y)
                self.accuracy = np.mean(predictions==self.y)
                if(self.accuracy > 99.0):
                    break
                print(f' Training Loss after a iteration {i}:{loss} || Accuracy: {self.accuracy * 100}')
                
    def bgd_with_l2(self,X, y, learning_rate= 0.01, iteration = 10000):
        self.X = X
        self.y = y
        self.loss_list = []
        self.acc_list = []
        for i in range(iteration):
            self.forward_pass(self.X)
            predictions = np.argmax(self.output3_act, axis=1)
            gradient_output3_act         = self.softmax_categorical_cross_entropy_combined_backward(self.output3_act, self.y)
#             print(gradient_output3_act)
            gradient_output3, gradient_input3= self.linear_backward_with_l2(self.output2,self.weights3,gradient_output3_act)
            gradient_output2_act      = self.ReLU_backward(gradient_input3, self.output2)
            gradient_output2, gradient_input2 = self.linear_backward_with_l2(self.output1, self.weights2, gradient_output2_act)
            gradient_output1_act            = self.ReLU_backward(gradient_input2, self.output1)
            gradient_output1, gradient_input1 = self.linear_backward_with_l2(self.X, self.weights1, gradient_output1_act)
            
            self.weights3  = self.weights3 - learning_rate * gradient_output3
            self.weights2  = self.weights2 - learning_rate * gradient_output2
            self.weights1  = self.weights1 - learning_rate * gradient_output1
            assert np.sum(gradient_output1) != np.nan, "The gradient has nan"
            assert np.sum(gradient_output1) != np.inf, "The gradient has inf"
            loss = self.compute_loss_with_l2(self.output3_act, y)
            self.accuracy = np.mean(predictions==self.y)
            if i%100 == 0:
                self.loss_list.append(loss)
                self.acc_list.append(self.accuracy)
                if(self.accuracy > .99):
                    break
                print(f'Loss after a iteration {i}:{loss} || Accuracy: {self.accuracy * 100}')
        plt.plot(self.loss_list)
        plt.title("Training loss of the model")   
    
    def rand_mini_batches(self, X, Y, mini_batch_size = 128, seed=1):
   
        classes = Y.shape[0]
        np.random.seed(seed)            
        m = X.shape[0]                  # number of training examples
        mini_batches = []

    #     Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
    #     shuffled_Y = Y[:, permutation].reshape((classes,m))
        shuffled_Y = Y.iloc[permutation]

    #     Partition (shuffled_X, shuffled_Y) except for the last batch
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size 
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[ k * mini_batch_size : (k+1)*mini_batch_size, : ]
    #         mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y.iloc[k * mini_batch_size : (k+1)*mini_batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Last batch (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
    #         mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y = shuffled_Y.iloc[ num_complete_minibatches * mini_batch_size : m]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches   
    
    def mini_batch_gd(self, X, y, learning_rate= 0.01, iteration = 100, mini_batch_size = 128):
        self.X_train = X
        self.y_train = y
        self.loss_list = []
        self.acc_list = []
        for i in range(iteration):
            seed = iteration
            mini_batches= self.rand_mini_batches(X =  self.X_train, Y = self.y_train, mini_batch_size = mini_batch_size, seed=seed)
            for mini_batch in mini_batches:
                mini_batch_X , mini_batch_y = mini_batch
                self.forward_pass(mini_batch_X)
                predictions = np.argmax(self.output3_act, axis=1)
                gradient_output3_act                  = self.softmax_categorical_cross_entropy_combined_backward(self.output3_act, mini_batch_y)
    #             print(gradient_output3_act)
                gradient_output3, gradient_input3= self.linear_backward_with_l2(self.output2,self.weights3,gradient_output3_act)
                gradient_output2_act= self.ReLU_backward(gradient_input3, self.output2)
                gradient_output2, gradient_input2 = self.linear_backward_with_l2(self.output1, self.weights2, gradient_output2_act)
                gradient_output1_act= self.ReLU_backward(gradient_input2, self.output1)
                gradient_output1, gradient_input1= self.linear_backward_with_l2(mini_batch_X, self.weights1, gradient_output1_act)

                self.weights3  = self.weights3 - learning_rate * gradient_output3
                self.weights2  = self.weights2 - learning_rate * gradient_output2
                self.weights1  = self.weights1 - learning_rate * gradient_output1
                assert np.sum(gradient_output1) != np.nan, "The gradient has nan"
                assert np.sum(gradient_output1) != np.inf, "The gradient has inf"
                loss = self.compute_loss_with_l2(self.output3_act, mini_batch_y)
                self.accuracy = np.mean(predictions==mini_batch_y)
                predictions_test = self.predict(self.X_val)
                test_accuracy = np.mean(predictions_test == self.y_val)
            self.loss_list.append(loss)
            self.acc_list.append(self.accuracy)
            print(f'Epoch({i+1}): Training Loss :{"%.2f"%loss} || Training Accuracy: {"%.2f"%(self.accuracy * 100)}%|| Validation Accuracy :{"%.2f"%(test_accuracy * 100)}% ')
        plt.plot(self.loss_list)
        plt.title("Training loss of the model")
        
    def load_model(self, weights, biases = None):
        self.weights1 = weights['1']
        self.weights2 = weights['2']
        self.weights3 = weights['3']
        
        self.biases1 = weights['b1']
        self.biases2 = weights['b2']
        self.biases3 = weights['b3']
    
    def save_model(self, filename = f'model'):
        from datetime import date

        today = date.today()

        filename = f'{filename}_{self.accuracy}-{today}.pkl'
        weights = {
                    '1': self.weights1, '2': self.weights2, '3': self.weights3, 
                    'b1':self.biases1,'b2':self.biases2,'b3':self.biases3
        }
        pickle.dump(weights, open(filename, 'wb'))
    
    def fit(self, X,y, optimizer = 'bgd', iters = 10000, learning_rate = 0.001, batch_size = None, regularizer = 'l2'):
        '''
        A Method to fit the Model
        X<pd.DataFrame> : The set of features
        y<pd.DataFrame> : The target Labels
        optimizer<str> : The optimizer for the model (bgd, mini_batch)
        learning_rate<float>: The learning rate for the model
        batch_size<int>(optional): 1 results in SGD, optional when optimizer = 'bgd' 
        regularizer<str>(optional) :values(None/'l2') The regularizer for the system 
        
        '''
        if (optimizer == 'bgd' and  regularizer is None) :
            self.bgd(self, X, y, learning_rate= learning_rate, iteration = iteration)
        elif (optimizer == 'bgd') and (regularizer == 'l2'):
            self.bgd_with_l2(X = X, y = y, learning_rate = learning_rate, iteration = iteration)
        elif (optimizer == 'mini_batch'):
            if batch_size is not None:
                self.mini_batch_gd(X = X, y= y, learning_rate= learning_rate, iteration = iters, mini_batch_size = batch_size)
            else: 
                raise Exception("Specify the batch_size for Mini Batch Gradient Descent")
        else:
            raise Exception("Invalid parameters passed !")
            
    def get_validation_data(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        
    def predict(self, X_test):
        wt1, wt2, wt3 = self.weights1.copy(), self.weights2.copy(), self.weights3.copy()
        b1, b2, b3    = self.biases1.copy(), self.biases2.copy(), self.biases3.copy()
        output1     = self.forward(X_test, wt1, b1)
        output1_act = self.ReLU(output1)
        output2     = self.forward(output1_act,  wt2, b2)
        output2_act = self.ReLU(output2)
        output3     = self.forward(output2_act,  wt3, b3)
        output3_act = self.Softmax(output3)
        prediction, prediction_prob = np.argmax(output3_act, axis=1), np.max(output3_act, axis=1)
        return prediction
    
    def predict_proba(self, X_test):
        wt1, wt2, wt3 = self.weights1.copy(), self.weights2.copy(), self.weights3.copy()
        b1, b2, b3    = self.biases1.copy(), self.biases2.copy(), self.biases3.copy()
        output1       = self.forward(X_test, wt1, b1)
        output1_act   = self.ReLU(output1)
        output2       = self.forward(output1_act,  wt2, b2)
        output2_act   = self.ReLU(output2)
        output3       = self.forward(output2_act,  wt3, b3)
        output3_act   = self.Softmax(output3)
        prediction, prediction_prob = np.argmax(output3_act, axis=1), np.max(output3_act, axis=1)
        output3_selected = output3_act
        values = []
        for i in range(0,3):
            prediction_i, prediction_prob_i= np.argmax(output3_selected, axis=1), np.max(output3_selected, axis=1)
            output3_selected[0][prediction_i] = -1
            values.append([prediction_i[0], prediction_prob_i[0]])
        return prediction


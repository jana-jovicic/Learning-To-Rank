from dataPoint import *
import random
import copy
import time
import numpy as np

'''
Simplified implementation of RankNet using papers:
1) http://research.microsoft.com/en-us/um/people/cburges/papers/icml_ranking.pdf
2) https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjHoo_C7cvlAhXh0qYKHVN7Ca4QFjAAegQIBBAC&url=http%3A%2F%2Fciteseerx.ist.psu.edu%2Fviewdoc%2Fdownload%3Fdoi%3D10.1.1.180.634%26rep%3Drep1%26type%3Dpdf&usg=AOvVaw3i__R1DtXUq242X_Gp3EjQ
'''

# Activation function of neurons. In this case, sigmoid function.
def activationFunc(x):
    return (1.0/(1.0+np.exp(-x)))

#The derivative of the activation function
def activationFuncDerivative(x):
    return np.exp(-x)/(pow(np.exp(-x)+1,2))

def random_float(low,high):
    return random.random()*(high-low) + low


def extractPairsOfDataPoints(dataPoints):
    pairs = []
    for i in range(0, len(dataPoints)-1):
        for j in range(i+1, len(dataPoints)):

            # We are only looking for documents with same id
            if dataPoints[i].qid != dataPoints[j].qid:
                break
            
            # If documents have same id, we are looking for ones with different rating
            if dataPoints[i].y != dataPoints[j].y:
                if(dataPoints[i].y > dataPoints[j].y):
                    pairs.append([dataPoints[i], dataPoints[j]])
                else:
                    pairs.append([dataPoints[j], dataPoints[i]])                    
    return pairs


class RankNet():
    
    def __init__(self, dataPoints, numInputs, numHidden, learningRate=0.001):
        self.dataPoints = dataPoints
        
        self.numInputs = numInputs + 1  # Number of input nodes (+1 is for bias node: A node with a constant input of 1 which is used to shift 
                                        # the transfer function.)
        self.numHidden = numHidden      # Number of hidden nodes
        self.numOutput = 1              # Number of output nodes. Assuming single output.

        # Current activation levels for nodes (the nodes' output value)
        self.activations_input = [1.0]*self.numInputs
        self.activations_hidden = [1.0]*self.numHidden
        self.activation_output = 1.0
        self.learning_rate = learningRate

        #A matrix with all weights from input layer to hidden layer
        self.weights_input = np.zeros((self.numInputs,self.numHidden))
        #A list with all weights from hidden layer to the output neuron.
        self.weights_output = np.zeros(self.numHidden)
        # set them to random vaules
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                self.weights_input[i][j] = random_float(-0.5, 0.5)
        for j in range(self.numHidden):
            self.weights_output[j] = random_float(-0.5, 0.5)
            
        #Data for the backpropagation step.
        #For storing the previous activation levels of all neurons
        self.prevInputActivations = []
        self.prevHiddenActivations = []
        self.prevOutputActivation = 0
        #For storing the previous delta in the output and hidden layer
        self.prevDeltaOutput = 0
        self.prevDeltaHidden = np.zeros(self.numHidden)
        
        #For storing the current delta in the same layers
        self.deltaOutput = 0
        self.deltaHidden = np.zeros(self.numHidden)

    
    def propagate(self, features):
        
        if len(features) != self.numInputs-1:
            raise ValueError('Wrong number of inputs')

        # input activations
        self.prevInputActivations=copy.deepcopy(self.activations_input)
        for i in range(self.numInputs-1):
            self.activations_input[i] = features[i]
        self.activations_input[-1] = 1 #Set bias node to -1.

        # hidden activations
        self.prevHiddenActivations=copy.deepcopy(self.activations_hidden)
        for j in range(self.numHidden):
            sum = 0.0
            for i in range(self.numInputs):
                sum += self.activations_input[i] * self.weights_input[i][j]
            self.activations_hidden[j] = activationFunc(sum)

        # output activations
        self.prevOutputActivation=self.activation_output
        sum = 0.0
        for j in range(self.numHidden):
            sum += self.activations_hidden[j] * self.weights_output[j]
        self.activation_output = activationFunc(sum)
        
        #print(self.activation_output)
        return self.activation_output
    

    def backpropagate(self):
        '''
        Backward propagation of error
        1. Compute delta for all weights going from the hidden layer to output layer (Backward pass)
        2. Compute delta for all weights going from the input layer to the hidden layer (Backward pass continued)
        3. Update network weights
        '''
        self.computeOutputDelta()
        self.computeHiddenDelta()
        self.updateWeights()
        
        
    def computeOutputDelta(self):
        
        # Equations [1-3] in 1)        
        # In prevOutputActivation is result from propagation of first document in pair,
        # In activation_output is result from propagation of secont document in pair
        # probability that document i should be ranked higher than document j via a sigmoid function:
        Pij = 1/(1 + np.exp(-(self.prevOutputActivation - self.activation_output)))
        self.prevDeltaOutput = activationFuncDerivative(self.prevOutputActivation)*(1.0-Pij)
        self.deltaOutput = activationFuncDerivative(self.activation_output)*(1.0-Pij)
        
    def computeHiddenDelta(self):
        
        #Equations [4-5] in 1)
           
        #Update delta_{I}
        for i in range(self.numHidden):
            self.prevDeltaHidden[i] = activationFuncDerivative(self.prevHiddenActivations[i])*self.weights_output[i]*(self.prevDeltaOutput-self.deltaOutput)
        #Update delta_{J}
        for j in range(self.numHidden):
            self.deltaHidden[j] = activationFuncDerivative(self.activations_hidden[j])*self.weights_output[j]*(self.prevDeltaOutput-self.deltaOutput)


    def updateWeights(self):
        
        # Update weights going from the input layer to the output layer.
        
        # Each input node is connected with all nodes in the hidden layer
        for j in range(self.numHidden):
            for i in range(self.numInputs):
                self.weights_input[i][j] += self.learning_rate*(self.prevDeltaHidden[j]*self.prevInputActivations[i]-self.deltaHidden[j]*self.activations_input[i])
                
        #Update weights going from the hidden layer (i) to the output layer (j)
        for i in range(self.numHidden):
            self.weights_output[i] += self.learning_rate*(self.prevDeltaOutput*self.prevHiddenActivations[i]-self.deltaOutput*self.activations_hidden[i])
    
    
    def countMisorderedPairs(self, pairs):
        
        '''
        https://medium.com/@nikhilbd/intuitive-explanation-of-learning-to-rank-and-ranknet-lambdarank-and-lambdamart-fe1e17fac418
        The cost function for RankNet aims to minimize the number of inversions in ranking. Here an inversion means an incorrect order among a pair of results, i.e. when we rank a lower rated result above a higher rated result in a ranked list.
        This function counts how many times the network makes the wrong judgement.
        errorRate = numWrong/(Total)
        '''
        misorderedPairs = 0

        for pair in pairs:
            self.propagate(pair[0].featureVector)
            self.propagate(pair[1].featureVector)
            if self.prevOutputActivation <= self.activation_output:
                misorderedPairs += 1
        
        return misorderedPairs / float(len(pairs))


    def train(self, iterations = 20):

        pairs = extractPairsOfDataPoints(self.dataPoints)
        
        errorRate = []
        startTime = time.time()
        
        for it in range(iterations):
            print("*** Epoch {0} ***" .format(it+1))
            for pair in pairs:
                self.propagate(pair[0].featureVector)
                self.propagate(pair[1].featureVector)
                self.backpropagate()   
            errorRate.append(self.countMisorderedPairs(pairs))
            print ('Error rate: %.2f' %errorRate[it])          
           
        m, s = divmod(time.time() - startTime, 60)
        print("Training took %dm %.1fs" %(m, s))
        
        #for w in self.weights_input:
            #print(w, end = ' ')
        #print("\n")
        #for w in self.weights_output:
            #print(w, end = ' ')
        #print("\n")

        # Save model
        np.save('model_RankNet', np.array([self.weights_input, self.weights_output]))
        
        
        
    def test(self):
        
        model = np.load("model_RankNet.npy", allow_pickle=True)
        self.weights_input = model[0]
        self.weights_output = model[1]
        pairs = extractPairsOfDataPoints(self.dataPoints)
        print("loss: {0}". format(self.countMisorderedPairs(pairs)))
        
        
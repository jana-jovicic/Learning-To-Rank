import numpy as np
import random

'''
Implemented using paper:
https://papers.nips.cc/paper/2023-pranking-with-ranking.pdf
'''

def maxY(dataPoints):
    k = -np.inf
    for dp in dataPoints:
        if dp.y > k:
            k = dp.y
    return k


class PRank():
    
    def __init__(self, dataPoints, iters=1000):
        self.dataPoints = dataPoints
        self.iters = iters
        self.k = maxY(dataPoints)
        self.bs = np.zeros(self.k)  
        self.bs[-1] = np.inf
        
        # In Microsoft dataset all feature vectors are the same length (missing values are noted as 0),
        # so for length of vector w, we can use the length of first feature vector
        self.ws = np.zeros([len(dataPoints[0].featureVector)])
        
        
    def train(self):
        
        numOfDataPoints = len(self.dataPoints)
        loss = 0
        random.seed()
        
        for t in range(self.iters):
        
            # Get a new rank-value x^t from R^n
            n = random.randint(0, len(self.dataPoints))-1
            xs = np.array(self.dataPoints[n].featureVector)
            y_real = self.dataPoints[n].y
            
            # predict y_pred = min{r : w^t * x^t - b[r]^t < 0}
            boolArr = self.bs > np.dot(xs, self.ws)     # e.g. array([False,  True,  True])
            y_pred = np.where(boolArr == True)[0][0]    # filters only True values, and gets first value of filtered array, which is the minimal r for which w^t * x^t - b[r]^t < 0
            
            #print("real={0} pred={1}".format(y_real, y_pred))
            
            '''
            alternative:
            y_pred = np.argmax(self.bs > np.dot(xs, self.ws))
            e.g. [0,np.inf,np.inf] > np.dot([1,1,1],[1,2,3])  -->  array([False,  True,  True])
                  np.argmax([0,np.inf,np.inf] > np.dot([1,1,1],[1,2,3]))    --> 1  (It returns first index that satisfies condition. 
                                                                                    In this case: min r for which bs[r] > np.dot(x,w))
            '''
            
            loss += abs(y_real - y_pred)
            
            if y_pred != y_real:
                ys = np.zeros(self.k-1)
                
                for r in range (self.k-1):
                    
                    if y_real <= r:
                        ys[r] = -1
                    else:
                        ys[r] = 1   
                        
                    ts = np.zeros(self.k-1)
                    if (np.dot(self.ws, xs) - self.bs[r])*ys[r] <= 0:
                        ts[r] = ys[r]
                    else:
                        ts[r] = 0
                        
                # Update w
                self.ws += sum(ts)*xs
                
                # Update b
                for r in range (self.k-1):
                    self.bs[r] -= ts[r]
                    
                    
        return self.ws, self.bs, 1.0*loss/self.iters
        #return 1.0*loss/self.iters
        
        
        
    def test(self):
        
 #       with open("model_prank.txt", "r") as f:
            
        model = np.load("model_prank.npy", allow_pickle=True)
        ws = model[0]
        bs = model[1]
        
        #for w in ws:
            #print("{0}".format(w), end = " ")
        #print("\n")
        #for b in bs:
            #print("{0}".format(b), end = " ")
            
        correctPredictions = 0
        
        for dp in self.dataPoints:
            
            boolArr = bs > np.dot(dp.featureVector, ws)
            y_pred = np.where(boolArr == True)[0][0]
            
            if y_pred == dp.y:
                correctPredictions += 1
                
        accuracy = correctPredictions / len(self.dataPoints)
        print("Accuracy = {0}".format(accuracy))
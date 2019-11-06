import sys
from prank import *
from rankNet import *
from dataPoint import *
    

def main():
    
    if len(sys.argv) != 3:
        print("Usage: python3 train.py rankingModel relPathToTrainigFile\n rankingModel = PRank | RankNet")
        sys.exit(0)
        
    print("Reading data...")
    dataPoints = read_data(sys.argv[2]) 
    
    #for dp in dataPoints:
        #dp.printDataPoint()
        
    if sys.argv[1] == "PRank":
        prank = PRank(dataPoints, 1000)
        print("Training...")
        ws, bs, loss = prank.train()
        print("loss = {0}".format(loss))
        np.save('model_prank', np.array([ws, bs]))
        
        #f = open("model_prank.txt", "w+")
        #f.write("{0} {1}".format(ws, bs))
        #f.close()
        
    elif sys.argv[1] == "RankNet":
        # number of input nodes is equal to number of features in feature vector
        rankNet = RankNet(dataPoints, len(dataPoints[0].featureVector), 20, 0.001)
        print("Training...")
        rankNet.train()

if __name__ == "__main__":
    main()
import sys
from dataPoint import *
from rankNet import *
from prank import *


def main():
    
    if len(sys.argv) != 3:
        print("Usage: python3 test.py rankingModel relPathToTestFile\n rankingModel = PRank | RankNet")
        sys.exit(0)
        
    print("Reading data...")
    dataPoints = read_data(sys.argv[2]) 
        
    if sys.argv[1] == "PRank":
        prank = PRank(dataPoints)
        print("Testing...")
        prank.test()
    elif sys.argv[1] == "RankNet":
        rankNet = RankNet(dataPoints, len(dataPoints[0].featureVector), 20, 0.001)
        print("Testing...")
        rankNet.test()
        

if __name__ == "__main__":
    main()
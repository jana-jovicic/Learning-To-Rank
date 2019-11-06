

def read_data(data_path):
    
    dataPoints = []
    
    with open(data_path, "r") as f:

        for line in f:
            if not line:
                break
            if "#" in line:  # ignore the comment
                line = line[:line.index("#")]
                
            splits = line.strip().split(" ")
            
            y = int(splits[0])
            qid = int(splits[1].split(":")[1])
            featureVector = [float(split.split(":")[1]) for split in splits[2:]]
            
            dp = DataPoint(y, qid, featureVector)
            
            dataPoints.append(dp)
                        
    return dataPoints



class DataPoint:
    
    def __init__ (self, y, qid, featureVector):
        self.y = y      # label
        self.qid = qid  # num in qid:num
        self.featureVector = featureVector
    
    def printDataPoint(self):
        print("y={0}  qid={1}  fv=".format(self.y, self.qid), end=' ')
        for fv in self.featureVector:
            print(fv, end=' ')
        print()

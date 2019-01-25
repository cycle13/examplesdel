import os
import numpy as np

def loadDataSet(fileName):
    xArr = []
    yArr = []
    fp = open(fileName, 'r')
    for line in fp:
        #print(line)
        #print(line[0:2])
        xArr.append(int(line[0:2]))
        yArr.append(float(line[2:4]))
        #print(xArr.append)
        #print(yArr.append)
        #os.system("pause")#debug
    return xArr,yArr
	
if __name__ == '__main__':
    # 加载数据集
    time_arr ,car_arr= loadDataSet('old.txt')
    print('tim_arr = \n',time_arr)
    print('car_arr = \n',car_arr)
	

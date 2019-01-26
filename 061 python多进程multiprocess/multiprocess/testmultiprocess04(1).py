#coding=utf-8
import logging
import time
from multiprocessing import Pool
logging.basicConfig(level=logging.INFO, filename='logger.log')
class Point:
  def __init__(self, x = 0, y= 0):
    self.x = x
    self.y = y
  def __str__(self):
    return "(%d, %d)" % (self.x, self.y)
def fun1(point):
  point.x = point.x + 3
  point.y = point.y + 3
  time.sleep(1)
  return point
def fun2(x):
  time.sleep(1)
  logging.info(time.ctime() + ", fun2 input x:" + str(x))
  return x * x
if __name__ == '__main__':
  pool = Pool(4)
  #test1
  mylist = [x for x in range(10)]
  ret = pool.map(fun2, mylist)
  print ret
  #test2
  mydata = [Point(x, y) for x in range(3) for y in range(2)]
  res = pool.map(fun1, mydata)
  for i in res:
    print str(i)
  #end
  pool.close()
  pool.join()
  print "end"

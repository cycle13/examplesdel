# coding = utf-8
import os
import cv2
import time

dataset_dir = "./data"
dataset_out_dir = "./out"
def convert(in_dir,out_dir,width,height):
    if not os.path.exists(out_dir):
           os.makedirs(out_dir)
    file_list = os.listdir(in_dir)
    print(file_list)
    for filename in file_list:
        path = ''
        path = in_dir+'/'+filename
        ####################读入图像###############################
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)#以灰度图的方式读取图片 cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
        #void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR );
        res = cv2.resize(image,(width,height), interpolation=cv2.INTER_LANCZOS4) #
		#INTER_NEAREST - 最邻近插值
        #INTER_LINEAR - 双线性插值，如果最后一个参数你不指定，默认使用这种方法
        #INTER_AREA - resampling using pixel area relation.
        #INTER_CUBIC - 4x4像素邻域内的双立方插值
        #INTER_LANCZOS4 - 8x8像素邻域内的Lanczos插值
        ####################写入图像########################		
        path = out_dir+'/'+filename
        cv2.imwrite(path,res)        
        print ("%s has been resized!"%filename)

if __name__ == '__main__':
   time1 = time.time()
   file_list = os.listdir(dataset_dir)
   print(file_list)
   for dirname in file_list:
        in_path = ''
        in_path = dataset_dir+'/'+dirname
        print (in_path)
        convert(in_path,dataset_out_dir+'/'+dirname+'_50x50',50,50)
   time2=time.time()
   print (u'总共耗时：' + str(time2 - time1) + 's')

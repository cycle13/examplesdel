# coding = utf-8
import os
import cv2
import time

dataset_dir = "./data"
dataset_out_dir = "./out"
size = 3#大图改大3，小图改小1
fat = 5#大图改大5，小图改小3
txtstr = 'elesun1986'

def puttxtonimg(in_dir,out_dir):
    if not os.path.exists(out_dir):
           os.makedirs(out_dir)
    file_list = os.listdir(in_dir)
    print(file_list)
    for filename in file_list:
        path = ''
        path = in_dir+'/'+filename
        file_attribute = os.stat(path)
        filemt= time.localtime(file_attribute.st_mtime)#文件创建时间
        txtstr = time.strftime("%Y-%m-%d %H:%M",filemt)
        print (txtstr)
		####################读入图像###############################
        image = cv2.imread(path,cv2.IMREAD_COLOR)#以rgb的方式读取图片 cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
        #print ('image.shape:',image.shape)
        #print ('image.shape:',image.shape[0])
        #txt = cv2.putText(image,'elesun',(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),1)
        txt = cv2.putText(image,txtstr,(int(image.shape[1]-size*20*len(txtstr)),int(image.shape[0]-size*20)),\
							cv2.FONT_HERSHEY_SIMPLEX,size,(0,255,255),fat)
		#各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
		#cv2.FONT_HERSHEY_SIMPLEX
		#cv2.FONT_HERSHEY_PLAIN
		#cv2.FONT_HERSHEY_DUPLEX
		#cv2.FONT_HERSHEY_COMPLEX
		#cv2.FONT_HERSHEY_TRIPLEX
		#cv2.FONT_HERSHEY_COMPLEX_SMALL
		#cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
		#cv2.FONT_HERSHEY_SCRIPT_COMPLEX
		#cv2.FONT_ITALIC
		#字体大小，数值越大，字体越大
		#字体粗细，越大越粗，数值表示描绘的线条占有的直径像素个数
        ####################写入图像########################		
        path = out_dir+'/'+filename
        cv2.imwrite(path,txt)        
        print ("%s has been texted!"%filename)

if __name__ == '__main__':
   time1 = time.time()
   file_list = os.listdir(dataset_dir)
   print(file_list)
   for dirname in file_list:
        in_path = ''
        in_path = dataset_dir+'/'+dirname
        print (in_path)
        puttxtonimg(in_path,dataset_out_dir+'/'+dirname+'_txt')
   time2=time.time()
   print (u'总共耗时：' + str(time2 - time1) + 's')

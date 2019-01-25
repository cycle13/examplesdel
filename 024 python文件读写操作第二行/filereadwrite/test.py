# coding: utf-8

filename = 'a.csv'
filename1 = 'b.csv'
f = open(filename,'r')
f1 = open(filename1,'w') # 若是'wb'就表示写二进制文件

next(f) # 从第二行开始读取数据
for line in f:
    print(line)
    f1.write(line)
f.close()
f1.close()

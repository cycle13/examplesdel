# coding: utf-8

filename = 'a.csv'
filename1 = 'b.csv'
f = open(filename,'r')
f1 = open(filename1,'w') # ����'wb'�ͱ�ʾд�������ļ�

next(f) # �ӵڶ��п�ʼ��ȡ����
for line in f:
    print(line)
    f1.write(line)
f.close()
f1.close()

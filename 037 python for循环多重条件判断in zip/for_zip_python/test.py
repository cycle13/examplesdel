print('**********example1********')
list1 = [2,3,4]
list2 = [4,5,6]
 
for x,y in zip(list1,list2):
    print('x=',x,'y=',y,'x*y=',x*y)

	
print('**********example2********') 	
list1 = [2,3,4]
list2 = [4,5,6,7]

for x,y in zip(list1,list2):
    print('x=',x,'y=',y,'x*y=',x*y)

print('**********example3********') 	
list1 = [2,3,4]

for x,n in zip(list1,range(5)):
    print('x=',x,'n=',n)
	
print('**********example4********') 	
list1 = [2,3,4]

for n,x in zip(range(5),list1):
    print('x=',x,'n=',n)
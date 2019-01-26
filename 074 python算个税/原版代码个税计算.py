# 固定赋值部分
money = [0, 3000, 12000, 25000, 35000, 55000, 80000, 99999999]
rate = [0, 0.03, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45]
deduct = [0, 0, 210, 1410, 2660, 4410, 7610, 15160]

def get_revenue_value(num):
	revenue = 0
	for i in range(len(money)):
		if num >= money[i] and num <= money[i+1]:
			revenue += num * rate[i+1] - deduct[i+1]
			break
	return revenue

def revenue_estimate(husincome, wifincome, huscd, wifcd, childnum, house):
	husrev = max(husincome-sum(huscd)-5000, 0)
	wifrev = max(wifincome-sum(wifcd)-5000, 0)
	childedu = [i*500 for i in range(childnum*2+1)]
	husadlist = [i+j for i in childedu for j in [0, house]]
	print(husadlist)
	result = [99999, 0, 0]
	for i in range(len(husadlist)):
		husad = husadlist[i]
		wifad = 1000 * childnum + house - husad
		husrevtest = husrev - husad
		wifrevtest = wifrev - wifad
		revenue = 0
		revenue = get_revenue_value(husrevtest) + get_revenue_value(wifrevtest)
		print(get_revenue_value(husrevtest), get_revenue_value(wifrevtest))  # 测试
		if revenue <= result[0]:
			result[0] = revenue
			result[1] = husad
			result[2] = wifad
		print(husad, wifad, revenue)  # 测试
		print('-------------------')  # 测试

	return result



if __name__ == '__main__':
	# 双方收入
	husincome = 5000
	wifincome = 5000
	# 继续教育和赡养老人抵扣额
	huscd = [400, 2000]
	wifcd = [400, 2000]
	# 受教育子女数量和租房/房贷抵扣额
	childnum = 2
	house = 1500
	result = revenue_estimate(husincome, wifincome, huscd, wifcd, childnum, house)
	# print('男方养老、继续教育抵扣金额{}元，子女教育、住房抵扣金额{}元'.format(sum(huscd), result[1]))
	# print('女方养老、继续教育抵扣金额{}元，子女教育、住房抵扣金额{}元'.format(sum(wifcd), result[2]))
	# print('最低税额{}元'.format(result[0]))

	print (result)


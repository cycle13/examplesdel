# -*- coding: utf-8 -*-
'''
incometax

env:Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
'''
# 固定赋值部分
threshold = 5000
money = [0, 3000, 12000, 25000, 35000, 55000, 80000, 99999999]#income range
rate = [0, 0.03, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45]#tax rate
deduct = [0, 0, 210, 1410, 2660, 4410, 7610, 15160]#susuan kouchu

def calulate_income_rank(num):
	for id in range(1,len(money)):
		if num > money[id-1] and num <= money[id]:
			break
		elif num <= 0 :
			id = 0
			break
	return id

if __name__ == '__main__':
	print('*************************************************************')
	#income 双方月收入名义工资（含基本工资+绩效+补助，税前）
	income_hus = 18000.0
	income_wif = 5900.0
	print('-------------------------------------------------------------')
	print('男方月收入名义工资为：{}元，女方月收入名义工资为：{}元'.format(income_hus, income_wif))
	
	
	#secure_fund 双方社保公积金 
	#社保有的按照最低标准缴费，有的是按照工资比例缴费（输入个人缴纳额即可）
	secure_fund_hus = 1440.0 + 1497.6 #(income_hus*0.104) + (income_hus*0.1)
	secure_fund_wif = 1245.92 + 608.0 #
	print('-------------------------------------------------------------')
	print('男方社保公积金扣除为：{}元，女方社保公积金扣除为：{}元'.format(secure_fund_hus, secure_fund_wif))	
	
	
	#welfare fine 双方扣费罚款等
	#迟到、请假、未转正扣除
	welfare_fine_hus = 662.07
	welfare_fine_wif = 169.54
	print('-------------------------------------------------------------')
	print('男方扣费罚款扣除为：{}元，女方扣费罚款扣除为：{}元'.format(welfare_fine_hus, welfare_fine_wif))
	
	
	#welfare_fine 双方专项附加扣除
	#家庭租房房贷、家庭子女教育（子女暂时计算在男方）、个人继续教育、个人父母养老、大病医疗（暂时不计）
	print('-------------------------------------------------------------')
	# 家庭租房房贷、家庭子女教育（子女暂时计算在男方）、个人继续教育、个人父母养老、大病医疗（暂时不计）
	additional_deduction_hus = [1500, 1000, 400, 1000, 0]
	additional_deduction_wif = [   0,    0,   0, 2000, 0]
	print('||男方租房房贷为：{}元，女方租房房贷为：{}元||'.format(additional_deduction_hus[0], additional_deduction_wif[0]))
	print('||男方子女教育为：{}元，女方子女教育为：{}元||'.format(additional_deduction_hus[1], additional_deduction_wif[1]))
	print('||男方继续教育为：{}元，女方继续教育为：{}元||'.format(additional_deduction_hus[2], additional_deduction_wif[2]))
	print('||男方父母养老为：{}元，女方父母养老为：{}元||'.format(additional_deduction_hus[3], additional_deduction_wif[3]))
	print('||男方大病医疗为：{}元，女方大病医疗为：{}元||'.format(additional_deduction_hus[4], additional_deduction_wif[4]))
	additional_deduction_hus = sum(additional_deduction_hus)
	additional_deduction_wif = sum(additional_deduction_wif)
	print('男方专项附加扣除为：{}元，女方专项附加扣除为：{}元'.format(additional_deduction_hus, additional_deduction_wif))
	
	#amount_before_tax 双方个人所得税前应纳所得额
	#个人所得税前应纳所得额=月收入名义工资-社保公积金-补贴扣费-专项附加扣除
	amount_before_tax_hus = income_hus - secure_fund_hus - welfare_fine_hus - additional_deduction_hus - threshold
	amount_before_tax_wif = income_wif - secure_fund_wif - welfare_fine_wif - additional_deduction_wif - threshold
	print('-------------------------------------------------------------')
	print('男方税前应纳所得额为：{}元，女方税前应纳所得额为：{}元'.format(amount_before_tax_hus, amount_before_tax_wif))
	
	#personal_tax 双方个人所得税应缴
	#个人所得税应缴=(个人所得税前应纳所得额 - 个税起征点)*对应税率-对应数算扣除数
	i = calulate_income_rank(amount_before_tax_hus)
	j = calulate_income_rank(amount_before_tax_wif)
	#print('测试个人所得税率为：{}，测试个人所得税率为：{}'.format(rate[calulate_income_rank(-500)], rate[calulate_income_rank(100000)]))#test
	if (amount_before_tax_hus > 0) :
		personal_tax_hus = amount_before_tax_hus*rate[i] - deduct[i]
	else :
		personal_tax_hus = 0
	if (amount_before_tax_wif > 0) :
		personal_tax_wif = amount_before_tax_wif*rate[j] - deduct[j]
	else :
		personal_tax_wif = 0
	print('-------------------------------------------------------------')
	print('男方个人所得税率为：{}，女方个人所得税率为：{}'.format(rate[i], rate[j]))
	print('男方个人所得税应缴为：{}元，女方个人所得税应缴为：{}元'.format(personal_tax_hus, personal_tax_wif))
	
	#total_deduction 双方应扣合计
	#应扣合计 = 社保公积金 + 罚款扣费 + 应缴个人所得税
	total_deduction_hus = secure_fund_hus + welfare_fine_hus + personal_tax_hus
	total_deduction_wif = secure_fund_wif + welfare_fine_wif + personal_tax_wif
	print('-------------------------------------------------------------')
	print('男方实际应扣合计为：{}元，女方实际应扣合计为：{}元'.format(total_deduction_hus, total_deduction_wif))
	
	#income_after_tax 双方实际收入
	#实际收入 = 月收入名义工资 - 社保公积金 - 补贴扣费等 - 个人所得税应缴
	income_after_tax_hus = income_hus - total_deduction_hus
	income_after_tax_wif = income_wif - total_deduction_wif
	print('-------------------------------------------------------------')
	print('男方实际收入合计为：{}元，女方实际收入合计为：{}元'.format(income_after_tax_hus, income_after_tax_wif))

	


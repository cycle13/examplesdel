ssh -o ServerAliveInterval=50 sunfengzhen@140.143.26.167
密码：sunfz123456

$scp -r ./cifar10_checkpoint_keras/  sunfengzhen@140.143.26.167:/home/sunfengzhen/
cod: sunfz123456

$ nohup python cifar10_checkpoint.py >out.log &

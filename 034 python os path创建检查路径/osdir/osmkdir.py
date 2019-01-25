
import os

model_dir = 'model'
model_name = 'model_train_radar.h5'
out_dir = 'output/01'

model_file = os.path.join(model_dir,'model_train_radar.h5')

print (os.getcwd())
print (model_file)

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if os.path.isfile(model_file):
    print("model_file exist")
else:
    print("model_file not exist")

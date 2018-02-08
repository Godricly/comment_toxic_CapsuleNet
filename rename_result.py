import shutil
kfold=10
rename = 2
for i in range(kfold):
    # result_path = 'data/result'+str(i)+'.csv'
    result_path = 'result'+str(i)+'.csv'
    result_new_path = 'result'+str(rename) + str(i)+'.csv'
    shutil.move(result_path, result_new_path)
    


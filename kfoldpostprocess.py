import pandas as pd
import numpy as np
from preprocess import get_label


kfold=5
for i in range(kfold):
    result_path = 'result'+str(i)+'.csv'
    result = pd.read_csv(result_path)
    pred_label = get_label(result)
    if i==0:
        mean_result = pred_label
    else:
        mean_result += pred_label

mean_result /= kfold
labels = ['toxic', 'severe_toxic',
          'obscene', 'threat',
          'insult', 'identity_hate']
result[labels] = mean_result
result.to_csv('result_kfold.csv', index=False)

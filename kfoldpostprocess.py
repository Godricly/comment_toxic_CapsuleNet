import pandas as pd
import numpy as np
from preprocess import get_label

kfold=30
for i in range(kfold):
    result_path = 'data/result'+str(i)+'.csv'
    # result_path = 'result'+str(i)+'.csv'
    result = pd.read_csv(result_path)
    pred_label = get_label(result)
    if i==0:
        mean_result = pred_label
    else:
        # mean_result *= pred_label
        mean_result += pred_label

# mean_result = np.power(mean_result, 1.0/kfold)
mean_result = mean_result / kfold

labels = ['toxic', 'severe_toxic',
          'obscene', 'threat',
          'insult', 'identity_hate']
result[labels] = mean_result
result.to_csv('result_kfold.csv', index=False)

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4
mean_result **= PROBABILITIES_NORMALIZE_COEFFICIENT
# mean_result =np.log(mean_result)
# mean_result -=0.5
# mean_result =np.exp(mean_result)
result[labels] = mean_result
result.to_csv('postprocessing1.csv', index=False)

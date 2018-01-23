import pandas as pd
import numpy as np
from preprocess import get_label

result_path = 'result.csv'
result = pd.read_csv(result_path)
pred_label = get_label(result)
pred_label[:,0] = np.max(pred_label, axis=1)
result['toxic'] = pred_label[:,0]
result.to_csv('result_post.csv', index=False, float_format='%.3f')

import pandas as pd
import numpy as np
from preprocess import get_label

result_path = 'result.csv'
result = pd.read_csv(result_path)
labels = ['toxic', 'severe_toxic',
          'obscene', 'threat',
          'insult', 'identity_hate']
mean_result = get_label(result)
PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4
mean_result **= PROBABILITIES_NORMALIZE_COEFFICIENT
result[labels] = mean_result
result.to_csv('postprocessing1.csv', index=False)

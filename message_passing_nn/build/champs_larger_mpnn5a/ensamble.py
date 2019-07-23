import os
import numpy as np
import pandas as pd
from common import get_path, get_data_path

def run_ensamble():
    ensemble_file = get_path() + 'data/submission/ensamble/submit-ensemble-JHC-all_types3.csv'

    df_test = pd.read_csv(get_data_path() + 'test.csv')            

    df_test = df_test.sort_values(by=['id'])

    test_id   = df_test.id.values
    test_type = df_test.type.values

    num_test=len(test_id)
    coupling_count = np.zeros(num_test)
    coupling_value = np.zeros(num_test)

    subs = [(['3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH'],
              get_path() + 'data/submission/zzz/submit/submit-00325000_model-larger.csv'),
            (['1JHC', '2JHC', '3JHC'],
              get_path() + 'data/submission/all_JHC/submit/submit-00237500_model-larger.csv')]

    for valid_type, f in subs:
            df = pd.read_csv(f)
            df = df.sort_values(by=['id'])
            for t in valid_type:
                index = np.where(test_type==t)[0]
                coupling_value[index] += df.scalar_coupling_constant.values[index]
                coupling_count[index] += 1
                
    coupling_value = coupling_value/coupling_count

    df = pd.DataFrame(list(zip(test_id, coupling_value)), columns=['id', 'scalar_coupling_constant'])
    df.to_csv(ensemble_file, index=False)

# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_ensamble()

    asdf1 = pd.read_csv(get_path() + 'data/submission/ensamble/submit-ensemble-JHC-all_types3.csv')
    asdf2 = pd.read_csv(get_path() + 'data/submission/zzz/submit/submit-00325000_model-larger.csv')

    print(asdf1.describe())
    print(asdf2.describe())
    print(asdf1.scalar_coupling_constant.mean()-asdf2.scalar_coupling_constant.mean())

    print('\nsuccess!')

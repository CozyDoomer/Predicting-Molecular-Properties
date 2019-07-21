import os
import numpy as np
import pandas as pd
from common import get_path

def run_ensamble():
    ensemble_file = get_path() + 'data/subs/blends/submit-ensemble-larger-00.csv'

    df_test = pd.read_csv(get_path() + 'data/test.csv'))               

    df_test = df_test.sort_values(by=['id'])

    id   = df_test.id.values
    type = df_test.type.values

    num_test=len(id)
    coupling_count = np.zeros(num_test)
    coupling_value = np.zeros(num_test)

    subs = [(['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '2JHH', '3JHH',],
              get_path() + 'data/submission/zzz/submit-00325000_model-larger.csv'),
            (['3JHN'],
              get_path() + 'data/submission/3JHN/submit-00000000_model.csv'),
            (['1JHC', '2JHC', '3JHC', ],
              get_path() + 'data/submission/all_JHC/submit-00000000_model.csv')]

    for valid_type, f in subs:
            df = pd.read_csv(f)
            df = df.sort_values(by=['id'])
            for t in valid_type:
                index = np.where(type==t)[0]
                coupling_value[index] += df.scalar_coupling_constant.values[index]
                coupling_count[index] += 1


    coupling_value = coupling_value/coupling_count


    df = pd.DataFrame(list(zip(id, coupling_value)), columns =['id', 'scalar_coupling_constant'])
    df.to_csv(ensemble_file,index=False)

# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_ensamble()

    print('\nsuccess!')

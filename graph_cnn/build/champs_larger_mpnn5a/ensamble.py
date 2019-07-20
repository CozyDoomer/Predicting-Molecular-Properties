import os
import numpy as np
import pandas as pd

def run_ensamble():
    ensemble_file = ('/run/media/windisk/Users/chrun/Documents/Projects/'
                     'Predicting-Molecular-Properties/graph_cnn/data/subs/blends/submit-ensemble-larger-00.csv')

    df_test = pd.read_csv(('/run/media/windisk/Users/chrun/Documents/Projects/'
                           'Predicting-Molecular-Properties/data/test.csv'))                

    df_test = df_test.sort_values(by=['id'])

    id   = df_test.id.values
    type = df_test.type.values

    num_test=len(id)
    coupling_count = np.zeros(num_test)
    coupling_value = np.zeros(num_test)

    ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH',]

    #TODO add own submissions
    #
    # mpnn   0.00010  197.5*  49.4 |  -0.702, - 1.431,  -1.122,  -0.648,  -1.683,  -1.761,  -1.625,  -1.467   | -1.213  0.30 -1.30 | -1.231 |  3 hr 53 min
    # lightgbm                        -0.3981, -1.7055, -0.9291, -1.6048, -1.2033, -1.5529, -0.9857, -1.8029


    subs = [(['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '2JHH', '3JHH',],
              '/root/share/project/kaggle/2019/champs_scalar/result/ensemble/submit-00370000_model-larger--add-poola.csv'),
            (['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '2JHH', '3JHH',],
              '/root/share/project/kaggle/2019/champs_scalar/result/separate/all/submit/submit-00327500_model.csv'),
            (['3JHN'],
              '/root/share/project/kaggle/2019/champs_scalar/result/separate/3jhn-00/submit/submit-00327500_model.csv'),
            (['1JHC', '2JHC', '3JHC', ],
              '/root/share/project/kaggle/2019/champs_scalar/result/separate/all-jhc-00/submit/submit-00262500_model.csv')]

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

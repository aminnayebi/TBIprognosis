import numpy as np
from model import Connection, Regression, IMPACT_Reg, RNNGRUD, RNNModel, ANN
import sys


if __name__ == '__main__':
    model_type = sys.argv[1]
    output_type = sys.argv[2]
    imput_method = sys.argv[3]
    max_len = int(sys.argv[4])
    num_neurons = int(sys.argv[5])
    num_hidden = int(sys.argv[6])
    epochs = int(sys.argv[7])
    total_fold = int(sys.argv[8])
    current_fold = int(sys.argv[9])

    c = Connection(verbose=False)
    c.clean_clinic_data(miss_rate=0.2)
    
    #---TEMPORALLY ADDED--------
    c.clean_gcs_data()
    c.clean_vital_data()
    c.clean_lab_data()
    c.lab_data.drop(['DLHematocrit'], axis=1, inplace=True)
    c.gcs_data.drop(['GCSEye_1-No Response','GCSEye_2-To Pain', 'GCSEye_3-To Verbal Command',
                    'GCSEye_4-Spontaneously', 'GCSEye_S-Untestable (Swollen)',
                    'GCSVrb_1-No Response','GCSVrb_2-Incomprehensible Sounds',
                    'GCSVrb_3-Inappropriate Words','GCSVrb_4-Disoriented & Converses',
                    'GCSVrb_5-Oriented & Converses','GCSVrb_T-Untestable (Artificial Airway)'],
                  axis=1, inplace=True)
    df = c.time_series()
    #-------------------------------------------

    if model_type == 'allReg':
        model = Regression(c, imput_method=imput_method, total_fold=total_fold, current_fold=current_fold)
    elif model_type == 'impact':
        model = IMPACT_Reg(c, output_type=output_type, imput_method=imput_method, total_fold=total_fold, current_fold=current_fold)
    elif model_type == 'grud':
        model = RNNGRUD(c, output_type=output_type, imput_method=imput_method, max_len=max_len, num_hidden=num_hidden,
                        num_neurons=num_neurons, epochs=epochs, total_fold=total_fold, current_fold=current_fold,
                        drop_rates=[0.421, 0.584, 0.297, 0.297], reg_rate=0.208, es_patience=49, learning_rate=0.00024)
    elif model_type == 'rnn':
        model = RNNModel(c, output_type=output_type, imput_method=imput_method, max_len=max_len, num_hidden=num_hidden,
                        num_neurons=num_neurons, epochs=epochs, total_fold=total_fold, current_fold=current_fold,
                        drop_rates=[0.421, 0.584, 0.297, 0.297], reg_rate=0.208, es_patience=49, learning_rate=0.00024)
    elif model_type == 'ann':
        model = ANN(c, output_type=output_type, imput_method=imput_method,
                    epochs=epochs, total_fold=total_fold, current_fold=current_fold)

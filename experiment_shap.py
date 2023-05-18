# SET RANDOM SEEDS-------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------




import pandas as pd
import numpy as np
from util import *
from preprocess import *
from model import RNNModel
from tensorflow.keras import backend as K
import shap
from shap_util import *
from copy import deepcopy
import timeit
from memory_profiler import memory_usage

import sys
import argparse

def main(exp_id, test_id, cv):
    model_type = 'grud'
    exp_df = pd.read_csv(DATA_PATH + 'Shap Experiments/experiment features.csv')
    exp_df = exp_df[exp_df.id==exp_id]
    
    print('Experiment ID = {}, Test ID = {}, and CV= {}'.format(exp_id, test_id, cv))
    print(exp_df)

    

    ### DATA PREPARATION and MODEL--------------------
    c = Connection(verbose=False)
    c.clean_clinic_data(miss_rate=0.2, max_gcs=15)
    c.clean_gcs_data()
    c.clean_vital_data()
    c.clean_lab_data()

    K.clear_session()
    rnn = RNNModel(c, output_type='Binary', imput_method='multivar', max_len=120, num_hidden=50,
                  num_neurons=100, epochs=2000, total_fold=10, current_fold=cv, save_bool=False,
                  drop_rates=[0.421, 0.584, 0.297, 0.297], reg_rate=0.208, es_patience=80,
                  learning_rate=0.00024)

    predicted_y = rnn.model.predict([rnn.test_x[i][test_id:test_id+1] for i in range(3)])
    true_y = rnn.test_y[test_id:test_id+1]
    print('prediction is :{}'.format(predicted_y))
    
    ### SHAP Methods-------------------------------
    model = rnn.model
    method = exp_df.method.iloc[0]
    num_background = int(exp_df.num_background)
    background_ts_1, background_mask_1 = rnn.train_x[0][:num_background], rnn.train_x[1][:num_background]
    test_ts_1, test_mask_1 = rnn.test_x[0][test_id:test_id + 1], rnn.test_x[1][test_id:test_id + 1]
    
    tic = timeit.default_timer()
    if method == 'STW':
        window_len = int(exp_df.window_len)
        stw = StationaryTimeWindow(model, window_len, B_ts=background_ts_1, B_mask=background_mask_1,
                                   test_ts=test_ts_1, test_mask=test_mask_1, model_type=model_type)
        ts_phi = stw.shap_values()
    elif method == 'SLTW':
        window_len = int(exp_df.window_len)
        stride = int(exp_df.stride)
        sltw = SlidingTimeWindow(model, stride, window_len, B_ts=background_ts_1, B_mask=background_mask_1, 
                                 test_ts=test_ts_1, test_mask=test_mask_1, model_type=model_type)
        ts_phi = sltw.shap_values()
    elif method == 'BTW':
        delta = float(exp_df.delta)
        n_w = int(exp_df.n_w)
        btw = BinaryTimeWindow(model, delta, n_w, B_ts=background_ts_1, B_mask=background_mask_1,
                               test_ts=test_ts_1, test_mask=test_mask_1, model_type=model_type)
        ts_phi = btw.shap_values()
    elif method == 'TIMESHAP':
        top_feat = int(exp_df.top_feat)
        top_event = int(exp_df.top_event)
        tol = float(exp_df.tol)
        def f(x):
            n = x.shape[0]
            new_test_mask = np.tile(test_mask_1, (n,1,1))
            new_tsetp = np.tile(rnn.train_x[2][:1], (n,1,1))
            return model.predict([x, new_test_mask, new_tsetp])
        
        ts_phi = timeshap_to_array(f, rnn.train_x[0], rnn.test_x[0], test_ts_1,
                                   tol=tol, top_x_events=top_event, top_x_feats=top_feat)
    toc = timeit.default_timer()
    print('Total time: {}'.format(toc-tic))
    
    # Evaluation of the results
    max_p=95
    min_p=10
    max_seq_len = round(ts_phi.shape[1]/3)
    min_seq_len=1
    prtb_eval_time = np.zeros((max_p-min_p))
    seq_eval_time = np.zeros((max_p-min_p, max_seq_len-min_seq_len))
    prtb_eval_all = np.zeros((max_p-min_p))
    seq_eval_all = np.zeros((max_p-min_p, max_seq_len-min_seq_len))
    test_data = [rnn.test_x[i][test_id:test_id+1] for i in range(3)]
    for p in range(min_p, max_p):
        prtb_eval_time[p - min_p] = xai_eval_fnc(model, ts_phi, test_data, model_type=model_type,
                                                 eval_type='prtb', percentile=p, by='time')
        prtb_eval_all[p - min_p] = xai_eval_fnc(model, ts_phi, test_data, model_type=model_type,
                                                eval_type='prtb', percentile=p, by='all')
        for seq_len in range(min_seq_len, max_seq_len):
            seq_eval_time[p-min_p, seq_len-min_seq_len] = xai_eval_fnc(model, ts_phi, test_data,
                                                                       model_type=model_type, eval_type='sqnc_eval', 
                                                                       seq_len=seq_len, percentile=p, by='time')
            seq_eval_all[p-min_p, seq_len-min_seq_len] = xai_eval_fnc(model, ts_phi, test_data,
                                                                      model_type=model_type, eval_type='sqnc_eval', 
                                                                      seq_len=seq_len, percentile=p, by='all')
    

    ### SAVE FILE--------------------------------
    save_name = 'track_exp{}_test{}_cv{}'.format(exp_id, test_id, cv)
    with open(OUTPUT_PATH + save_name, 'wb') as f:
        saveObject = {'ts_phi':ts_phi,
                      'prtb_eval_time':prtb_eval_time,
                      'seq_eval_time':seq_eval_time,
                      'prtb_eval_all':prtb_eval_all,
                      'seq_eval_all':seq_eval_all,
                      'total_time':toc-tic,
                      'predicted_y': predicted_y,
                      'true_y': true_y}
        pickle.dump(saveObject, f)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shap Experiments')
    parser.add_argument('--index', type=int)
    parser.add_argument('--num_tests', type=int)
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--memory', type=bool, default=False)
    args = parser.parse_args()
    
    index = int(args.index) # Must start from 1 to total_experiments * num_tests
    num_tests = int(args.num_tests)
    cv = int(args.cv)
    
    # Calculating the exp_id and test sample id
    exp_id = int((index - 1)/num_tests) + 1
    test_id = index % num_tests
    
    if args.memory:
        print('******** Memory mode is enabled **********')
        mem = max(memory_usage((main, (exp_id, test_id, cv))))

        # Read already saved result
        save_name = 'track_exp{}_test{}_cv{}'.format(exp_id, test_id, cv)
        with open(OUTPUT_PATH + save_name, 'rb') as f:
            saveObject = pickle.load(f)
        # Change result file and save it again
        saveObject['memory']=mem
        with open(OUTPUT_PATH + save_name, 'wb') as f:
            pickle.dump(saveObject, f)
    else:
        main(exp_id, test_id, cv)


    
    
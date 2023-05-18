import unittest
import numpy as np

from shap_util import xai_eval_fnc

class Model():
    def predict(self, x):
        num_step = x.shape[1]
        num_feature = x.shape[2]
        idx = np.arange(num_step)
        idx = idx[..., np.newaxis]
        x = x*idx
        return np.sum(x)

class XaiEvalTestCase(unittest.TestCase):
    def test_time_wise_prtb(self):
        model = Model()
        percentile = 85
        
        input_ts = np.array([[[1,2,3,4,5,6,7,8,9,10],
                              [11,12,13,14,15,16,17,18,19,20]]])
        input_ts = input_ts.transpose((0,2,1))
        
        ts_phi = np.array([[[6,10,8,9,7,5,4,3,2,1],
                              [20,19,18,17,16,15,14,13,12,11]]])/20
        ts_phi = ts_phi.transpose((0,2,1))
        
        prtb_input_ts = np.array([[[1,8,3,6,5,6,7,8,9,10],
                                   [9,8,13,14,15,16,17,18,19,20]]])
        prtb_input_ts = prtb_input_ts.transpose((0,2,1))
        
        org_output = model.predict(prtb_input_ts)
        new_output = xai_eval_fnc(model, ts_phi, input_ts, model_type='lstm', percentile=percentile,
                                  eval_type='prtb', by='time')
        self.assertEqual(org_output, new_output)
    
    def test_time_wise_sqnc(self):
        model = Model()
        percentile = 85
        sequence_length = 3
        
        input_ts = np.array([[[1,2,3,4,5,6,7,8,9,10],
                              [11,12,13,14,15,16,17,18,19,20]]])
        input_ts = input_ts.transpose((0,2,1))
        
        ts_phi = np.array([[[6,10,8,9,7,5,4,3,2,1],
                              [20,19,18,17,16,15,14,13,12,11]]])/20
        ts_phi = ts_phi.transpose((0,2,1))
        
        prtb_input_ts = np.array([[[1, 0, 0, 0, 0, 0,7,8,9,10],
                                   [ 0, 0, 0, 0,15,16,17,18,19,20]]])
        prtb_input_ts = prtb_input_ts.transpose((0,2,1))
        
        org_output = model.predict(prtb_input_ts)
        new_output = xai_eval_fnc(model, ts_phi, input_ts, model_type='lstm', percentile=percentile,
                                  eval_type='sqnc_eval', seq_len=sequence_length, by='time')
        self.assertEqual(org_output, new_output)
    
    def test_all_wise_prtb(self):
        model = Model()
        percentile = 75
        
        input_ts = np.array([[[1,2,3,4,5],
                              [11,12,13,14,15]]])
        input_ts = input_ts.transpose((0,2,1))
        
        ts_phi = np.array([[[6,10,18,9,7],
                              [4,9,21,17,16]]])
        ts_phi = ts_phi.transpose((0,2,1))
        
        prtb_input_ts = np.array([[[1,2,2,4,5],
                                   [11,12,2,1,15]]])
        prtb_input_ts = prtb_input_ts.transpose((0,2,1))
        
        org_output = model.predict(prtb_input_ts)
        new_output = xai_eval_fnc(model, ts_phi, input_ts, model_type='lstm', percentile=percentile,
                                  eval_type='prtb', by='all')
        self.assertEqual(org_output, new_output)
        
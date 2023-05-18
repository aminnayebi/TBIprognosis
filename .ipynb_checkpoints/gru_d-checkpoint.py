from keras.layers import Input, Dense, Concatenate, RNN, BatchNormalization, Dropout
from keras.layers import Layer
import keras.backend as K
from keras.layers.recurrent import _generate_dropout_mask
from keras.models import Model
from keras import initializers
import numpy as np
from tqdm import tqdm
import pandas as pd
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score


class GRUDCell(Layer):
    '''
    This GRU unit is based on Che et al. (2018) paper. This code is written by
    Sindhura Tipirneni.
    '''
    
    def __init__(self, input_dim, hidden_dim, dropout=0, recurrent_dropout=0, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_size = (self.hidden_dim, self.input_dim, self.input_dim)
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self.activation = K.tanh
        self.recurrent_activation = K.sigmoid
        self.kernel_initializer = initializers.VarianceScaling(mode='fan_avg', distribution='uniform')
        self.recurrent_initializer = initializers.Orthogonal()
        self.bias_initializer = initializers.Zeros()
        super(GRUDCell, self).__init__()
    
    def get_config(self):
        config = {'input_dim': self.input_dim, 'hidden_dim': self.hidden_dim,
                  'dropout': self.dropout, 'recurrent_dropout': self.recurrent_dropout}
        base_config = super(GRUDCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
         
    def build(self, input_shape):
        
        # W_z, W_r, W
        self.kernel = self.add_weight(
                shape=(self.input_dim, self.hidden_dim * 3),
                name='kernel',
                initializer=self.kernel_initializer,
            )
        self.kernel_z = self.kernel[:, :self.hidden_dim]
        self.kernel_r = self.kernel[:, self.hidden_dim:self.hidden_dim*2]
        self.kernel_h = self.kernel[:, self.hidden_dim*2:]
        
        # U_z, U_r, U
        self.recurrent_kernel = self.add_weight(
                shape=(self.hidden_dim, self.hidden_dim * 3),
                name='recurrent_kernel',
                initializer=self.recurrent_initializer
            )
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.hidden_dim]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.hidden_dim:self.hidden_dim*2]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.hidden_dim*2:]
        
        # V_z, V_r, V
        self.masking_kernel = self.add_weight(
                shape=(self.input_dim, self.hidden_dim * 3),
                name='masking_kernel',
                initializer=self.kernel_initializer
            )
        self.masking_kernel_z = self.masking_kernel[:, :self.hidden_dim]
        self.masking_kernel_r = self.masking_kernel[:, self.hidden_dim:self.hidden_dim*2]
        self.masking_kernel_h = self.masking_kernel[:, self.hidden_dim*2:]
        
        # b_r, b_z, b
        self.bias = self.add_weight(
                shape=(3*self.hidden_dim,),
                name='bias',
                initializer=self.bias_initializer
                )
        self.input_bias_z = self.bias[:self.hidden_dim]
        self.input_bias_r = self.bias[self.hidden_dim:2*self.hidden_dim]
        self.input_bias_h = self.bias[2*self.hidden_dim:]
        
        # W_gamma_x
        self.input_decay_kernel = self.add_weight(
                shape=(self.input_dim,),
                name='input_decay_kernel',
                initializer=self.kernel_initializer
            )
        
        # b_gamma_x
        self.input_decay_bias = self.add_weight(
                    shape=(self.input_dim,),
                    name='input_decay_bias',
                    initializer=self.bias_initializer
                )
        
        # W_gamma_h
        self.hidden_decay_kernel = self.add_weight(
                shape=(self.input_dim, self.hidden_dim),
                name='hidden_decay_kernel',
                initializer=self.kernel_initializer
            )
        
        # b_gamma_h
        self.hidden_decay_bias = self.add_weight(
                    shape=(self.hidden_dim,),
                    name='hidden_decay_bias',
                    initializer=self.bias_initializer
                )
        super(GRUDCell, self).build(input_shape)
        
    def call(self, inputs, states, training=None):
        # Get inputs and states.
        input_x = inputs[:, :self.input_dim]
        input_m = inputs[:, self.input_dim:-1]
        input_s = inputs[:, -1:]
        h_tm1, x_keep_tm1, s_prev_tm1 = states
        input_d = input_s - s_prev_tm1
        # Get dropout.
        if 0. < self.dropout < 1. and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(K.ones_like(input_x),
                                        self.dropout, training=training, count=3)
        if (0. < self.recurrent_dropout < 1. and self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(K.ones_like(h_tm1),
                            self.recurrent_dropout, training=training, count=3)
        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask
        # Compute decay.
        gamma_di = input_d * self.input_decay_kernel  # element wise mu
        gamma_di = K.bias_add(gamma_di, self.input_decay_bias)
        gamma_di = K.exp(-K.relu(gamma_di))
        gamma_dh = K.dot(input_d, self.hidden_decay_kernel)
        gamma_dh = K.bias_add(gamma_dh, self.hidden_decay_bias)
        gamma_dh = K.exp(-K.relu(gamma_dh))
        # Compute decayed input and hidden state.
        x_keep_t = K.switch(input_m, input_x, x_keep_tm1)
        x_t = K.switch(input_m, input_x, gamma_di * x_keep_t)
        h_tm1d = gamma_dh * h_tm1
        # Apply dropout.
        if 0. < self.dropout < 1.:
            x_z, x_r, x_h = x_t * dp_mask[0], x_t * dp_mask[1], x_t * dp_mask[2]
        else:
            x_z, x_r, x_h = x_t, x_t, x_t
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z, h_tm1_r, h_tm1_h = (h_tm1d * rec_dp_mask[0], h_tm1d * rec_dp_mask[1], h_tm1d * rec_dp_mask[2])
        else:
            h_tm1_z, h_tm1_r, h_tm1_h = h_tm1d, h_tm1d, h_tm1d
        # Get z_t, r_t, hh_t, h_t.
        z_t = K.dot(x_z, self.kernel_z) + K.dot(h_tm1_z, self.recurrent_kernel_z) + K.dot(input_m, self.masking_kernel_z) + self.input_bias_z
        r_t = K.dot(x_r, self.kernel_r) + K.dot(h_tm1_r, self.recurrent_kernel_r) + K.dot(input_m, self.masking_kernel_r) + self.input_bias_r
        hh_t = K.dot(x_h, self.kernel_h) + K.dot(r_t*h_tm1_h, self.recurrent_kernel_h) + K.dot(input_m, self.masking_kernel_h) + self.input_bias_h
        z_t = self.recurrent_activation(z_t)
        r_t = self.recurrent_activation(r_t)
        hh_t = self.activation(hh_t)
        h_t = z_t * h_tm1 + (1 - z_t) * hh_t
        # Return necessary arrays.
        s_keep_t = K.switch(input_m, K.tile(input_s, [1, self.input_dim]), s_prev_tm1)
        return h_t, [h_t, x_keep_t, s_keep_t]

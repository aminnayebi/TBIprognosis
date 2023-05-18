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
    
    def __init__(self, input_dim, hidden_dim, dropout=0, recurrent_dropout=0):
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
        

class GRUD():
    
    def __init__(self):
        self.data_dir = ''
        self.num_time_steps = 48
        self.data_mean = None
        self.data_std = None
        self.train_data = self.get_data(self.data_dir+'set-a.csv')
        self.valid_data = self.get_data(self.data_dir+'set-b.csv')
        self.test_data = self.get_data(self.data_dir+'set-c.csv')
        self.train()
        
    def build_model(self):
        val = Input(shape=(self.num_time_steps, self.num_vars))
        obs = Input(shape=(self.num_time_steps, self.num_vars))
        ts = Input(tensor=K.ones_like(val)[:,:,0:1])
        inp_conc = Concatenate()([val,obs,ts])
        rec = RNN(GRUDCell(33,49,recurrent_dropout=0.3))(inp_conc)
        rec = Dropout(0.2)(rec)
        rec = BatchNormalization()(rec)
        op = Dense(1, activation='sigmoid')(rec)
        self.model = Model([val,obs,ts], op)
        self.model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy',
                           weighted_metrics=['accuracy'])
        self.model.summary()
        
    def train(self):
        self.num_epochs = 1000
        self.num_bat_per_epoch = 500
        self.batch_size = 32
        x_train, x_obs, y_train = self.train_data
        num_pat = len(x_train)
        ratio = (self.train_data[1]==0).sum()/(self.train_data[1]==1).sum()
        valid_rocs = []
        test_rocs = []
        for i in range(5):
            self.build_model()
            best_valid_auroc = -1
            for e in range(self.num_epochs):
                loss = 0
                for b in tqdm(range(self.num_bat_per_epoch)):
                    rand_pat = np.random.choice(range(num_pat), self.batch_size, 
                                                replace=False)
                    loss += self.model.train_on_batch([x_train[rand_pat], x_obs[rand_pat]], 
                                        y_train[rand_pat], class_weight={0:1,1:ratio})[0]
                print ('Epoch',e,': Loss :',loss/(self.batch_size))
                train_auroc, valid_auroc, test_auroc = self.test()
                if valid_auroc>best_valid_auroc:
                    best_valid_auroc = valid_auroc
                    best_epoch = e
                    best_test_auroc = test_auroc
                print ('Best epoch:', best_epoch, '\n')
                if e >= best_epoch+10:
                    break
            valid_rocs.append(best_valid_auroc)
            test_rocs.append(best_test_auroc)
        print ('Valid AUROC:', np.mean(valid_rocs), np.std(valid_rocs))
        print ('Test AUROC:', np.mean(test_rocs), np.std(test_rocs))
    
    def test(self):
        y_pred = self.model.predict([self.train_data[0], self.train_data[1]])
        train_auroc = roc_auc_score(self.train_data[2], y_pred)
        print ('Train AUROC :', train_auroc)
        y_pred = self.model.predict([self.valid_data[0], self.valid_data[1]])
        valid_auroc = roc_auc_score(self.valid_data[2], y_pred)
        print ('Valid AUROC :', valid_auroc)
        y_pred = self.model.predict([self.test_data[0], self.test_data[1]])
        test_auroc = roc_auc_score(self.test_data[2], y_pred)
        print ('Test AUROC :', test_auroc)
        return (train_auroc, valid_auroc, test_auroc)
    
    def get_data(self, filename):
        # Read observations.
        df = pd.read_csv(filename)
        pats = list(set(df['pat_id']))
        varis = sorted(list(set(df['name'])))
        self.num_vars = len(varis)
        pat_to_ind = self.inv_list(pats)
        var_to_ind = self.inv_list(varis)
        values = np.zeros((len(pats),self.num_time_steps,self.num_vars))
        obs = np.zeros((len(pats),self.num_time_steps,self.num_vars))
        for row in tqdm(df.itertuples()):
            pind = pat_to_ind[getattr(row, 'pat_id')]
            vind = var_to_ind[getattr(row, 'name')]
            tstep = getattr(row, 'hour')
            values[pind, tstep, vind] = getattr(row, 'value')
            obs[pind, tstep, vind] = 1
        # Read outcomes.
        outcomes_df = df.groupby('pat_id').agg({'in_hosp_death':'first'}).reset_index()
        outcomes = np.zeros(len(pats))
        for row in outcomes_df.itertuples():
            pind = pat_to_ind[getattr(row, 'pat_id')]
            outcomes[pind] = getattr(row,  'in_hosp_death')
        # Normalize data.
        if self.data_mean is None:
            mvalues = np.ma.array(values, mask=1-obs)
            self.data_mean = np.array(mvalues.mean(axis=(0,1)))
            self.data_std = np.array(mvalues.std(axis=(0,1)))
        values = (values-self.data_mean.reshape((1,1,self.num_vars)))/self.data_std.reshape((1,1,self.num_vars))
        values = obs*values
        # Return necessary arrays.
        return (values, obs, outcomes)
    
    def inv_list(self, l):
        d = {}
        for i in range(len(l)):
            d[l[i]] = i
        return d
    
    
if __name__ == '__main__':
    obj = GRUD()

        
        
        
        
import matplotlib.pyplot as plt
import copy
import random
from util import evaluation
import numpy as np
import pandas as pd
from model import DATA_PATH, FIG_PATH
from preprocess import Connection


def permutation_interpret(ann, plot=True, var_nums=10, num_folds=10,
                          filename=DATA_PATH + 'time_series_data.csv'):
    # time series data frame
    df = pd.read_csv(filename)
    df = df[df.DLTimeSinceInj < ann.num_time_steps]
    df = df[df.Guid.isin(ann.guid)]

    # List of all variables and non-transformed variables
    ts_vars = sorted(list(set(df['variable'])))
    ts_unique_vars = list(set([i.split('_')[0] for i in ts_vars]))
    
    ts_changes = [0]*num_folds
    for i_fold in range(num_folds):
        vals = []
        for grp_i in range(len(ts_unique_vars)):
            new_x = copy.deepcopy(ann.test_x)
            for i in range(ann.test_x[0].shape[2]):
                if ts_unique_vars[grp_i] in ts_vars[i]:
                    a = new_x[0][:, :, i].flatten()
                    b = new_x[1][:, :, i].flatten()
                    ab = list(zip(a, b))
                    random.Random(grp_i).shuffle(ab)
                    a, b = zip(*ab)
                    a = np.array(a).reshape((ann.test_x[0].shape[:2]))
                    b = np.array(b).reshape((ann.test_x[0].shape[:2]))

                    new_x[0][:, :, i] = a
                    new_x[1][:, :, i] = b

            new_predict = ann.model.predict(new_x)
            vals.append(evaluation(ann.test_y, new_predict, 'AMSE', output_type='OrdinalMulticlass'))
        ts_changes[i_fold] = rnn[i_fold].evaluate(metric='AMSE')[1] - vals

    # amount of change in AMSE value for permutation of each variable
    assert len(vals) == len(ts_unique_vars)
    ts_changes = np.array(ts_changes)
    ts_changes = ts_changes.mean(axis=0)
    ts_changes, ts_unique_vars = zip(*sorted(zip(ts_changes, ts_unique_vars), reverse=False))
    ts_changes = -np.array(ts_changes)

    if plot:
        temp_var = list(ts_unique_vars[-var_nums:])
        # temp_var.reverse()
        plt.barh(y=temp_var, width=ts_changes[-var_nums:])
        plt.xticks(rotation=90)
        plt.xlabel('Reduction of AMSE')
        plt.tight_layout()
        plt.savefig(FIG_PATH + 'feature importance (permutation) for time series.pdf')


def plot_time_series(c, time_step=10, p_id='TBIKU122DXE'):
    sbp_t = c.vital_data.VitalsTimeSinceInj[(c.vital_data.Guid == p_id) & (~ c.vital_data.DVSBP.isna())].values
    temp_t = c.vital_data.VitalsTimeSinceInj[(c.vital_data.Guid == p_id) & (~ c.vital_data.DvTemp.isna())].values
    urea_t = c.lab_data.DLTimeSinceInj[(c.lab_data.Guid == p_id) & (~ c.lab_data.DLUrea.isna())].values

    sbp = ['Systolic Blood Pressure'] * len(sbp_t)
    temp = ['Temperature'] * len(temp_t)
    urea = ['Labs (Urea)'] * len(urea_t)

    # Plotting the values of a patient at each time
    t_max = max(sbp_t)
    t_min = 0
    flag = 'k'

    plt.plot(sbp_t, sbp, 'ko')
    plt.plot(temp_t, temp, 'ko')
    plt.plot(urea_t, urea, 'ko')
    plt.xlabel('Time (hours)')
    while t_min < t_max:
        #     plt.axvspan(t_min, t_min+time_step, facecolor=flag, alpha=0.2)
        t_min += time_step
        if flag == 'k':
            flag = 'w'
        else:
            flag = 'k'
    plt.savefig(FIG_PATH + 'time series before aggregation.png', bbox_inches='tight')
    plt.show()

    # Plotting the values after aggregation
    sbp_t = np.floor(sbp_t / time_step)
    temp_t = np.floor(temp_t / time_step)
    urea_t = np.floor(urea_t / time_step)
    t_max = max(sbp_t)
    t_min = 0
    flag = 'k'

    plt.plot(sbp_t, sbp, 'ko')
    plt.plot(temp_t, temp, 'ko')
    plt.plot(urea_t, urea, 'ko')
    plt.xlabel('Time steps (every {} hours)'.format(time_step))
    while t_min < t_max:
        plt.axvspan(t_min, t_min + 1, facecolor=flag, alpha=0.2)
        t_min += 1
        if flag == 'k':
            flag = 'w'
        else:
            flag = 'k'

    plt.savefig(FIG_PATH + 'time series after aggregation.png', bbox_inches='tight')
    plt.show()


def missingness_plot(c, filename=DATA_PATH + 'time_series_data.csv',
                     hours=120, save_file=False):
    # preparing the dataframe
    df = pd.read_csv(filename)
    df = df[(df.DLTimeSinceInj < hours) & (df.Guid.isin(c.included_cases.Guid))]
    df_count = df.groupby(['variable', 'Guid'])['DLTimeSinceInj'].apply(lambda x: len(x) / (max(x) + 1)).reset_index()
    df_count.rename(columns={'DLTimeSinceInj': 'Freq'}, inplace=True)
    df_count = df_count.groupby('variable', as_index=False).mean()
    df_count.Freq = (1 - df_count.Freq) * 100

    df_count['category'] = df_count.apply(lambda x: ('Labs' if x['variable'] in c.lab_data.columns else
                                                     'Vitals' if x['variable'] in c.vital_data.columns else
                                                     'GCS & Pupils'), axis=1)
    df_count = df_count.groupby(['category', 'variable']).sum()

    # Some functions needed for plotting
    from itertools import groupby

    def test_table():
        data_table = pd.DataFrame({'Room': ['Room A'] * 4 + ['Room B'] * 4,
                                   'Shelf': (['Shelf 1'] * 2 + ['Shelf 2'] * 2) * 2,
                                   'Staple': ['Milk', 'Water', 'Sugar', 'Honey', 'Wheat', 'Corn', 'Chicken', 'Cow'],
                                   'Quantity': [10, 20, 5, 6, 4, 7, 2, 1],
                                   'Ordered': np.random.randint(0, 10, 8)
                                   })
        return data_table

    def add_line(ax, xpos, ypos):
        line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                          transform=ax.transAxes, color='black')
        line.set_clip_on(False)
        ax.add_line(line)

    def label_len(my_index, level):
        labels = my_index.get_level_values(level)
        return [(k, sum(1 for i in g)) for k, g in groupby(labels)]

    def label_group_bar_table(ax, df):
        ypos = -.1
        scale = 1. / df.index.size
        for level in range(df.index.nlevels - 1):
            pos = 0
            for label, rpos in label_len(df.index, level):
                lxpos = (pos + .5 * rpos) * scale
                ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
                add_line(ax, pos * scale, ypos)
                pos += rpos
            add_line(ax, pos * scale, ypos)
            ypos -= .1

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df_count.plot(kind='bar', stacked=False, ax=fig.gca())
    ax.get_legend().remove()
    # Below 3 lines remove default labels
    labels = ['' for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    ax.set_xlabel('')
    label_group_bar_table(ax, df_count)
    fig.subplots_adjust(bottom=.1 * df.index.nlevels)
    plt.ylabel('Percentage of missingness')
    plt.xlabel('\n \n Time Series Variables')
    plt.savefig(FIG_PATH + 'Distribution of the number of measurements among different variables.png', bbox_inches='tight')

    if save_file: df_count.to_csv(DATA_PATH + "time series variable index and frequency.csv", index=True)

def heat_map(start, stop, x, shap_values, var_name='Feature 1', plot_type='bar', title=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import BoundaryNorm
    from textwrap import wrap
    import numpy as np; np.random.seed(1)
    
    ## ColorMap-------------------------
    # define the colormap
    cmap = plt.get_cmap('PuOr_r')

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize and forcing 0 to be part of the colorbar!
    bounds = np.arange(np.min(shap_values),np.max(shap_values),.005)
    idx=np.searchsorted(bounds,0)
    bounds=np.insert(bounds,idx,0)
    norm = BoundaryNorm(bounds, cmap.N)
    ##------------------------------------
    
    if title is None: title = '\n'.join(wrap('{} values and contribution scores'.format(var_name), width=40))
    
    if plot_type=='heat' or plot_type=='heat_abs':
        plt.rcParams["figure.figsize"] = 9,3
        if plot_type=='heat_abs':
            shap_values = np.absolute(shap_values)
            cmap = 'Reds'
        fig, ax1 = plt.subplots(sharex=True)
        extent = [start, stop, -2, 2]
        im1 = ax1.imshow(shap_values[np.newaxis, :], cmap=cmap, norm=norm, aspect="auto", extent=extent)
        ax1.set_yticks([])
        ax1.set_xlim(extent[0], extent[1])
        ax1.title.set_text(title)
        fig.colorbar(im1, ax=ax1, pad=0.1)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(start, stop), x, color='black')
    elif plot_type=='bar':
        plt.rcParams["figure.figsize"] = 8,3
        fig, ax1 = plt.subplots(sharex=True)
        mask1 = shap_values < 0
        mask2 = shap_values >= 0
        ax1.bar(np.arange(start, stop)[mask1], shap_values[mask1], color='blue', label='Negative Shapely values')
        ax1.bar(np.arange(start, stop)[mask2], shap_values[mask2], color='red', label='Positive Shapely values')
        ax1.set_title(title)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(start, stop), x, 'k-', label='Sequential data points')
        # legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    ax1.set_xlabel('Time steps')
    if plot_type=='bar': ax1.set_ylabel('Shapely values')
    ax2.set_ylabel(var_name + ' sequential values')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    c = Connection(verbose=False)
    c.clean_clinic_data(miss_rate=0.2, cleaned_version=False, max_gcs=15, ext_subjects=False)
    # c.clean_vital_data()
    # c.clean_lab_data()
    # c.clean_gcs_data()
    # plot_time_series(c)

    missingness_plot(c)

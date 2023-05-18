import count_sketch
import numpy as np
import pickle
import timeit
import sys

def count_sketch_BO_experiments(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100,
                                low_dim=2, high_dim=25, initial_n=20, ARD=False, box_size=None,
                                noise_var=0):

    result_obj = np.empty((0, total_itr+initial_n))
    elapsed = np.empty((0, total_itr + initial_n))
    result_s = np.empty((0, initial_n + total_itr, low_dim))
    result_f_s = np.empty((0, initial_n + total_itr, 1))
    result_high_s = np.empty((0, initial_n + total_itr, high_dim))

    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()

        temp_result, temp_elapsed, temp_s, temp_f_s, _, temp_high_s = count_sketch.RunMain(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                              total_itr=total_itr, func_type=test_func, s=None, ARD=ARD,
                                                                              box_size=box_size, noise_var=noise_var)

        result_obj = np.append(result_obj, temp_result, axis=0)
        elapsed = np.append(elapsed, temp_elapsed, axis=0)
        result_s = np.append(result_s, [temp_s], axis=0)
        result_f_s = np.append(result_f_s, [temp_f_s], axis=0)
        result_high_s = np.append(result_high_s, [temp_high_s], axis=0)

        stop = timeit.default_timer()

        print(i)
        print(stop - start)

        # Saving the results for Hartmann6 in a pickle
    if test_func == 'Rosenbrock':
        file_name = 'result/rosenbrock_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'Branin':
        file_name = 'result/branin_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'Hartmann6':
        file_name = 'result/hartmann6_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'StybTang':
        file_name = 'result/stybtang_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'WalkerSpeed':
        file_name = 'result/walkerspeed_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'MNIST':
        file_name = 'result/mnist_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'GRUD':
        file_name = 'result/grud_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    else:
        raise

    fileObject = open(file_name, 'wb')
    save_dict = {'f_s': result_f_s, 'high_s': result_high_s}
    pickle.dump(save_dict, fileObject)
    fileObject.close()


if __name__=='__main__':
    start_rep = int(sys.argv[2])
    stop_rep = int(sys.argv[3])
    test_func = sys.argv[4]
    total_iter = int(sys.argv[5])
    low_dim = int(sys.argv[6])
    high_dim = int(sys.argv[7])
    initial_n = int(sys.argv[8])
    variance = int(sys.argv[9])

    if sys.argv[1]=='HeSBO':
        count_sketch_BO_experiments(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, box_size=1, noise_var=variance)
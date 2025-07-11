from itertools import permutations
import copy
import math
import numpy as np
import random


# Input the matrix_size. Although the matrix in NAS_Bench_101 is 7*7, but after removing the 'input' and 'output',
# the size will be 5.
# Output all the exchange pairs.
def calculate_exchange_pair(matrix_size=5):
    # List all possible permutations
    all_possible = list(permutations(np.arange(matrix_size), matrix_size))     #  permutation指全排列  列出节点
    # (0,1,2, ... ,matrix_size) is the base list
    base_list = list(all_possible[0])
    all_exchange_pair = []
    # print(all_possible)
    for i in range(1, len(all_possible)):   # 除了base_list的全部
        exchange_pair = []
        current_list = list(all_possible[i])
        for j in range(matrix_size - 1):
            # (matrix_size -1) times exchange are enough, because if the front (matrix_size - 1) are in right order,
            # the final element is at right position naturally.
            # j is the first exchange index
            if current_list[j] != base_list[j]:
                exchange_index = current_list.index(base_list[j])     # base_list中的元素在current_list（第一次）出现的位置
                current_list[j], current_list[exchange_index] = current_list[exchange_index], current_list[j]
                exchange_pair.append([j, exchange_index])             # 和基准变动情况 譬如基准是[43210] 现在是[01234] 则pair[05] [14] [23]即对调位置索引
        all_exchange_pair.append(exchange_pair)

    return all_exchange_pair
    # # if you would like to know the number of different numbers of exchanges, you can try the following codes
    # # for matrix_size=5, the result is 'num1: 10, num2: 35, num3: 50, num4: 24'
    # exchange1, exchange2, exchange3, exchange4 = 0, 0, 0, 0
    # for pair in all_exchange_pair:
    #     length = len(pair)
    #     if length == 1:
    #         exchange1 += 1
    #     elif length == 2:
    #         exchange2 += 1
    #     elif length == 3:
    #         exchange3 += 1
    #     else:
    #         exchange4 += 1
    # print(f'num1: {exchange1}, num2: {exchange2}, num3: {exchange3}, num4: {exchange4}')


def create_new_metrics(matrix, module_integers, select_upper_tri=False, max_num=-1, InOut=False):
    all_possible_metrics = []
    metric_dict = {'module_adjacency': np.array(matrix), 'module_integers': module_integers}
    all_possible_metrics.append(metric_dict)
    assert len(matrix) == len(module_integers) + 2           #  整数无输入输出
    # because the input and output are removed
    if InOut:
        matrix_size = len(matrix)
    else:
        matrix_size = len(module_integers)
    if max_num == -1:
        max_num = math.factorial(matrix_size)                # 算阶乘
    all_exchange_pair = calculate_exchange_pair(matrix_size) # 找到节点所有排列组合，返回节点位置替换信息

    for pair in all_exchange_pair:
        matrix_copy = copy.deepcopy(matrix)
        module_integers_copy = copy.deepcopy(module_integers) # copy会跟着变，deepcopy不会跟着变
        matrix_copy = np.array(matrix_copy)
        for i in range(len(pair)):
            first_index = pair[i][0] + 1
            last_index = pair[i][1] + 1
            # exchange row and column
            matrix_copy[[first_index, last_index], :] = matrix_copy[[last_index, first_index], :]    # 交换两行
            matrix_copy[:, [first_index, last_index]] = matrix_copy[:, [last_index, first_index]]    # 交换两列
            module_integers_copy[first_index - 1], module_integers_copy[last_index - 1] = module_integers_copy[
                                                                                              last_index - 1], \
                                                                                          module_integers_copy[
                                                                                              first_index - 1]
        metric_dict = {'module_adjacency': matrix_copy, 'module_integers': module_integers_copy}
        if select_upper_tri:
            if isupper_tri(matrix_copy):
                # only upper tri matrix can be added
                all_possible_metrics.append(metric_dict)
        else:
            # add metrics including non-upper-tri-matrix
            all_possible_metrics.append(metric_dict)
    if max_num == -1:                                        # 选择所有生成的增强数据
        return all_possible_metrics
    else:
        return random.sample(all_possible_metrics, max_num)  # 选择一定数量的增强数据


def isupper_tri(matrix):
    upper_tri_matrix = np.triu(matrix)
    is_triu = (matrix == upper_tri_matrix).all()
    # if one element is not the same, is_triu = False
    return is_triu


if __name__ == '__main__':
    # matrix = [[0, 1, 1, 0, 0],
    #           [0, 0, 0, 0, 1],
    #           [0, 0, 0, 1, 0],
    #           [0, 0, 0, 0, 1],
    #           [0, 0, 0, 0, 0]]
    # module_integers = [1, 2, 3]

    matrix = [[0, 1, 1, 1, 0, 1, 0],  # input layer
              [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
              [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
              [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
              [0, 0, 0, 0, 0, 0, 0]]  # output layer
    module_integers = [1, 2, 2, 2, 3]

    all_possible = create_new_metrics(matrix, module_integers)
    print(all_possible)
    # print(len(all_possible))
    # index = 0
    # for i in range(len(all_possible)):
    #     if np.array_equal(matrix, all_possible[i]['module_adjacency']) and np.array_equal(module_integers, all_possible[i]['module_integers']):
    #         index += 1
    #         print(all_possible[i])
    #         print('yes')
    #         print(i)
    # print(index)
    # all_possible includes the original architecture
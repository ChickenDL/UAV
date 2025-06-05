'''
Descripttion: 
version: 1.0
Author: ZXL
Date: 2024-05-07 10:58:17
LastEditors: ZXL
LastEditTime: 2025-04-14 10:55:29
'''
import os
import numpy as np
import random
import copy
import pickle
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse
from RelativeNet_101 import RelativeNetwork, TrainDataset, TestDataset
from population import PopulationX
from individual import IndividualX
from nasbench.lib import model_spec as _model_spec
from get_data_from_101 import NASBench101, padding_zero_in_matrix
from data_augmentation_HAAP import create_new_metrics
from utils import population_log, write_best_individual, find_all_simple_paths, build_adj_matrix_from_paths, matrix2utl, utl2matrix

logging.basicConfig(filename='RUN.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

current_path = os.path.dirname(__file__)

ModelSpec = _model_spec.ModelSpec
nasbench101 = NASBench101()


def query_fitness_for_indi(query_indi:IndividualX):
    model_spec = ModelSpec(matrix=query_indi.indi['matrix'], ops=query_indi.indi['op_list'])
    _, computed_stat = nasbench101.get_info_by_model_spec(model_spec)
    final_valid_accuracy_list = []
    final_test_accuracy_list = []
    x = random.randint(0, 2)
    random_final_valid_accuracy = computed_stat[108][x]['final_validation_accuracy']
    for i in range(3):
        final_valid_accuracy_list.append(computed_stat[108][i]['final_validation_accuracy'])
        final_test_accuracy_list.append(computed_stat[108][i]['final_test_accuracy'])
    mean_final_valid_accuracy = np.mean(final_valid_accuracy_list)
    mean_final_test_accuracy = np.mean(final_test_accuracy_list)

    query_indi.mean_acc = mean_final_valid_accuracy
    query_indi.random_acc = random_final_valid_accuracy
    query_indi.test_mean_acc = mean_final_test_accuracy
    

def query_fitness_for_pops(query_pop: PopulationX):
    index_list = []
    for i, indi in enumerate(query_pop.pops):
        model_spec = ModelSpec(matrix=indi.indi['matrix'], ops=indi.indi['op_list'])
        _, computed_stat = nasbench101.get_info_by_model_spec(model_spec)
        index_list.append(nasbench101.get_index_by_model_spec(model_spec))
        final_valid_accuracy_list = []
        final_test_accuracy_list = []
        x = random.randint(0, 2)
        random_final_valid_accuracy = computed_stat[108][x]['final_validation_accuracy']
        for j in range(3):
            final_valid_accuracy_list.append(computed_stat[108][j]['final_validation_accuracy'])
            final_test_accuracy_list.append(computed_stat[108][j]['final_test_accuracy'])
        mean_final_valid_accuracy = np.mean(final_valid_accuracy_list)
        mean_final_test_accuracy = np.mean(final_test_accuracy_list)

        query_pop.pops[i].mean_acc = mean_final_valid_accuracy
        query_pop.pops[i].random_acc = random_final_valid_accuracy
        query_pop.pops[i].test_mean_acc = mean_final_test_accuracy
    
    return index_list
    

class Evolution():
    def __init__(self, pc=0.2, pm=0.9, m_num_matrix=1, m_num_op_list=1, population_size=20):
        self.pc = pc
        self.pm = pm
        self.m_num_matrix = m_num_matrix
        self.m_num_op_list = m_num_op_list
        self.population_size = population_size
        self.pops = PopulationX(population_size)


    def initialize_popualtion(self):
        print("initializing population with number {}...".format(self.population_size))
        self.pops = PopulationX(self.population_size, self.m_num_matrix, self.m_num_op_list)
        population_log(0, self.pops, current_path)
    

    def recombinate(self, pop_size) -> PopulationX:
        print('mutation and crossover...')
        offspring_list = []
        for _ in range(int(pop_size / 2)):
            # tournament selection
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            # crossover
            if random.random() < self.pc:
                offset1, offset2 = self.crossover(p1, p2)
            else:
                offset1 = copy.deepcopy(p1)
                offset2 = copy.deepcopy(p2)
            # mutation
            if random.random() < self.pm:
                offset1.mutation()
            if random.random() < self.pm:
                offset2.mutation()
            offspring_list.append(offset1)
            offspring_list.append(offset2)
        offspring_pops = PopulationX(0)
        offspring_pops.set_populations(offspring_list)

        return offspring_pops
    

    def crossover(self, p1: IndividualX, p2: IndividualX, utl_len=21, op_list_len=7):
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        p1.clear_state_info()
        p2.clear_state_info()
        utl1 = matrix2utl(p1.indi['matrix'])
        utl2 = matrix2utl(p2.indi['matrix'])
        retry_num = 0
        while True:
            retry_num += 1
            cross_point = random.randint(1, utl_len - 1)
            crossed_utl1 = np.hstack((utl1[:cross_point], utl2[cross_point:]))
            crossed_utl2 = np.hstack((utl2[:cross_point], utl1[cross_point:]))
            crossed_matrix1 = utl2matrix(crossed_utl1)
            crossed_matrix2 = utl2matrix(crossed_utl2)
            model_spec1 = ModelSpec(matrix=crossed_matrix1, ops=p1.indi['op_list'])
            model_spec2 = ModelSpec(matrix=crossed_matrix2, ops=p2.indi['op_list'])
            # considering the invalid spec
            if model_spec1.valid_spec and (np.sum(model_spec1.matrix) <= 9) and model_spec2.valid_spec and (
                np.sum(model_spec2.matrix) <= 9):
                break
            if retry_num > 20:
                # print('Crossover has tried for more than 20 times, but still get invalid spec.\n'
                #       'Give up this crossover and go on...')
                crossed_matrix1 = p1.indi['matrix']
                crossed_matrix2 = p2.indi['matrix']
                break
        p1.indi['matrix'] = crossed_matrix1
        p2.indi['matrix'] = crossed_matrix2
        op_list1 = p1.indi['op_list']
        op_list2 = p2.indi['op_list']
        op_list_cross_point = random.randint(1, op_list_len - 1)
        crossed_op_list1 = np.hstack((op_list1[:op_list_cross_point], op_list2[op_list_cross_point:]))
        crossed_op_list2 = np.hstack((op_list2[:op_list_cross_point], op_list1[op_list_cross_point:]))
        p1.indi['op_list'] = crossed_op_list1.tolist()
        p2.indi['op_list'] = crossed_op_list2.tolist()
            
        return p1, p2
    

    # def crossover(self, p1: IndividualX, p2: IndividualX):
    #     p1 = copy.deepcopy(p1)
    #     p2 = copy.deepcopy(p2)
    #     p1.clear_state_info()
    #     p2.clear_state_info()
    #     retry_num = 0
    #     while True:
    #         retry_num += 1
    #         paths1 = []
    #         paths2 = []
    #         op_list1 = []
    #         op_list2 = []
    #         paths1 = find_all_simple_paths(p1.indi['matrix'], 0, 6)
    #         paths2 = find_all_simple_paths(p2.indi['matrix'], 0, 6)
    #         paths = paths1 + paths2
    #         if retry_num > 20 or len(paths) == 0:
    #             # print('Crossover has tried for more than 20 times, but still get invalid spec.\n'
    #             #         'Give up this crossover and go on...')
    #             op_list1 = p1.indi['op_list']
    #             op_list2 = p2.indi['op_list']
    #             op_list_cross_point = random.randint(1, 6)
    #             crossed_op_list1 = np.hstack((op_list1[:op_list_cross_point], op_list2[op_list_cross_point:]))
    #             crossed_op_list2 = np.hstack((op_list2[:op_list_cross_point], op_list1[op_list_cross_point:]))
    #             p1.indi['op_list'] = crossed_op_list1.tolist()
    #             p2.indi['op_list'] = crossed_op_list2.tolist()
    #             break
    #         selected_index1 = random.sample(range(0, len(paths)), int(len(paths) / 2) + 1)
    #         selected_index2 = random.sample(range(0, len(paths)), int(len(paths) / 2) + 1)
    #         for index in selected_index1:
    #             paths1.append(paths[index])
    #         for index in selected_index2:
    #             paths2.append(paths[index])
    #         crossed_matrix1 = build_adj_matrix_from_paths(paths1, 7)
    #         crossed_matrix2 = build_adj_matrix_from_paths(paths2, 7)
    #         op_list1 = p1.indi['op_list']
    #         op_list2 = p2.indi['op_list']
    #         op_list_cross_point = random.randint(1, 6)
    #         crossed_op_list1 = np.hstack((op_list1[:op_list_cross_point], op_list2[op_list_cross_point:]))
    #         crossed_op_list2 = np.hstack((op_list2[:op_list_cross_point], op_list1[op_list_cross_point:]))
    #         crossed_op_list1, crossed_op_list2 = crossed_op_list1.tolist(), crossed_op_list2.tolist()

    #         model_spec1 = ModelSpec(matrix=crossed_matrix1, ops=crossed_op_list1)
    #         model_spec2 = ModelSpec(matrix=crossed_matrix2, ops=crossed_op_list2)
    #         # considering the invalid spec
    #         if model_spec1.valid_spec and (np.sum(model_spec1.matrix) <= 9) and model_spec2.valid_spec and (
    #             np.sum(model_spec2.matrix) <= 9):
    #             p1.indi['matrix'], p1.indi['op_list'] = crossed_matrix1, crossed_op_list1
    #             p2.indi['matrix'], p2.indi['op_list'] = crossed_matrix2, crossed_op_list2
    #             break
        
    #     return p1, p2


    def environmental_selection(self, gen, offspring_population: PopulationX, elitism=0.1, is_random=False):
        # environment selection from the current population and the offspring population
        # assert (self.pops.get_pop_size() == self.population_size)
        # assert (offspring_population.get_pop_size() == self.population_size)
        print('environmental selection...')
        elite_num = int(self.population_size * elitism)
        indi_list = self.pops.pops
        indi_list.extend(offspring_population.pops)
        # descending order
        if is_random:
            indi_list.sort(key=lambda x: x.random_acc, reverse=True)
        indi_list.sort(key=lambda x: x.mean_acc, reverse=True)
        elitism_list = indi_list[0:elite_num]
        left_list = indi_list[elite_num:]
        np.random.shuffle(left_list)
        
        for _ in range(self.population_size - elite_num):
            i1 = random.randint(0, len(left_list) - 1)
            i2 = random.randint(0, len(left_list) - 1)
            winner = self.selection(left_list[i1], left_list[i2], is_random)
            
            elitism_list.append(winner)

        self.pops.set_populations(elitism_list)
        # record each generation's population and best individual
        population_log(gen, self.pops, current_path)
        write_best_individual(gen, self.pops, current_path)


    def tournament_selection(self):
        ind1_id = random.randint(0, self.pops.get_pop_size() - 1)
        ind2_id = random.randint(0, self.pops.get_pop_size() - 1)
        ind1 = self.pops.get_individual(ind1_id)
        ind2 = self.pops.get_individual(ind2_id)
        winner = self.selection(ind1, ind2)

        return winner
    

    def selection(self, ind1, ind2, is_random=False):
        if is_random:
            if ind1.random_acc > ind2.random_acc:
                return ind1
            else:
                return ind2
        if ind1.mean_acc > ind2.mean_acc:
            return ind1
        else:
            return ind2


def arch2data(matrix, op_list, isomorphic=False):
    data_list = []
    model_spec = ModelSpec(matrix, op_list)
    pruned_matrix, pruned_op_list = model_spec.matrix, model_spec.ops
    padding_matrix, padding_op_list = padding_zero_in_matrix(pruned_matrix, pruned_op_list)
    if isomorphic:
        all_possible = create_new_metrics(padding_matrix, padding_op_list[1:-1])
        for j in range(len(all_possible)):
            matrix = []
            op_list = []
            op_list.append('input')
            op_list.extend(all_possible[j]['module_integers'])
            op_list.append('output')
            matrix = all_possible[j]['module_adjacency']
            oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
                                'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0], 'output':[0, 0, 0, 0, 0, 1]}
            x = [oper2feature[oper] for oper in op_list]
            x = torch.tensor(x, dtype=torch.float)
            indices = np.where(matrix == 1)
            indices = np.array(indices)
            edge_index = torch.tensor(indices, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
    
        return data_list
    else:
        matrix, op_list = padding_matrix, padding_op_list
        oper2feature = {'input': [1, 0, 0, 0, 0, 0], 'conv1x1-bn-relu': [0, 1, 0, 0, 0, 0], 'conv3x3-bn-relu': [0, 0, 1, 0, 0, 0], 
                                'maxpool3x3': [0, 0, 0, 1, 0, 0], 'null': [0, 0, 0, 0, 1, 0], 'output':[0, 0, 0, 0, 0, 1]}
        x = [oper2feature[oper] for oper in op_list]
        x = torch.tensor(x, dtype=torch.float)
        indices = np.where(matrix == 1)
        indices = np.array(indices)
        edge_index = torch.tensor(indices, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        return data


def sample_expansion(data_sampled, offspring):
    for indi in offspring.pops:
        arch = {}
        arch['matrix'] = indi.indi['matrix']
        arch['op_list'] = indi.indi['op_list']
        arch['acc'] = indi.mean_acc
        data_sampled.append(arch)

    return data_sampled


def refine_TripletNetwork(model, EPOCH, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in tqdm(range(EPOCH), desc='Training/Updating Triplenet Network'):
        total_loss = 0
        for data in train_loader:
            anchor, positive, negative = data
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        info = 'Epoch :  [{:03d}/{}]'.format(epoch+1, EPOCH) + '   |   loss :  {:08f}'.format(average_loss)
        logging.info(info)

    return model


def contrast_select(modelX, pops, generation, evalnum, device):
    modelX.eval()
    unique = list(set(generation.pops)) # deduplication of generated architectures
    print(len(unique))
    candidate = [indi for indi in unique if indi not in pops] # remove architectures that are the same as those in the parent population
    print(len(candidate))
    data_sampled = []
    for ind in candidate:
        sampled_arch = {}
        sampled_arch['arch'] = {}
        sampled_arch['arch']['matrix'] = ind.indi['matrix']
        sampled_arch['arch']['op_list'] = ind.indi['op_list']
        data_sampled.append(sampled_arch)
    dataset = TestDataset(data_sampled, False)
    loader = DataLoader(dataset=dataset, batch_size=len(candidate) - 1, shuffle=False)
    scores = []
    with torch.no_grad():
        for index, pairs in enumerate(loader):
            data1, data2 = pairs
            data1, data2 = data1.to(device), data2.to(device)
            outputs = modelX(data1, data2)
            _, predicted = torch.max(outputs, 1)
            score = predicted.sum().item()
            scores.append(score)

    indices = np.argsort(scores)
    top_indices = indices[-evalnum:]
    offspring_pops = [candidate[indice] for indice in top_indices]

    offspring = PopulationX(0)
    offspring.set_populations(offspring_pops)

    return offspring


def main(args):
    pc = args.pc
    pm = args.pm
    m_num_matrix = args.m_num_matrix
    m_num_op_list = args.m_num_operation
    popsize = args.popsize
    offsize = args.offsize
    elitism = args.elitism
    generations = args.gens
    evalnum = args.evalnum
    number = args.number

    epoch = args.epochs
    batch_size = args.batch_size
    threshold = args.threshold

    data_sampled = []
    # randomly generate architecture
    nasbench101 = NASBench101()
    number = args.number


    for i in range(number):
        data = {}
        data_arch = {}
        index = random.randint(0, 423623)
        fixed_stat, computed_stat = nasbench101.get_info_by_index(index)
        matrix = fixed_stat['module_adjacency']
        op_list = fixed_stat['module_operations']
        final_test_accuracy_list = []
        final_valid_accuracy_list = []
        x = random.randint(0, 2)
        random_acc = computed_stat[108][x]['final_validation_accuracy']
        for i in range(3):
            final_valid_accuracy_list.append(computed_stat[108][i]['final_validation_accuracy'])
            final_test_accuracy_list.append(computed_stat[108][i]['final_test_accuracy'])
        mean_final_valid_accuracy = np.mean(final_valid_accuracy_list)
        mean_final_test_accuracy = np.mean(final_test_accuracy_list)

        mean_acc = mean_final_valid_accuracy
        test_mean_acc = mean_final_test_accuracy

        data_arch['matrix'] = matrix
        data_arch['op_list'] = op_list
        data['arch'] = data_arch
        data['acc'] = mean_acc
        
        data_sampled.append(data)

        # initialize the population by sampling from the search space
        # if i < popsize:
        #     indi = IndividualX(m_num_matrix, m_num_op_list)
        #     model_spec = ModelSpec(matrix, op_list)
        #     pruned_matrix, pruned_op_list = model_spec.matrix, model_spec.ops
        #     padding_matrix, padding_op_list = padding_zero_in_matrix(pruned_matrix, pruned_op_list)
        #     indi.indi['matrix'] = padding_matrix
        #     indi.indi['op_list'] = ['conv1x1-bn-relu' if ops == 'null' else ops for ops in padding_op_list]
        #     indi.mean_acc = mean_acc
        #     indi.random_acc = random_acc
        #     indi.test_mean_acc = test_mean_acc
        #     init_pops.append(indi)


    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    # dataset = TrainDataset(data_sampled=data_sampled, augmentation=False)
    # train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    RelativeNet = RelativeNetwork(input_channels=6, hidden_channels=64, output_channels=2).to(device)


    optimizer = optim.Adam(RelativeNet.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # refine_TripletNetwork(TripletNet, epoch, train_loader, optimizer, criterion, device)
    save_path = os.path.join(current_path, 'predictor', 'NASBench101', 'RelativeNet-300.pt')
    RelativeNet.load_state_dict(torch.load(save_path))

    Evol = Evolution(pc, pm, m_num_matrix, m_num_op_list, popsize)
    query_fitness_for_pops(Evol.pops)
    population_log(0, Evol.pops, current_path)
    arg_index = Evol.pops.get_sorted_index_order_by_acc()
    best = Evol.pops.get_individual(arg_index[0]) 

    select_dict = {}
    best_list = []

    # only evolve
    # for gen in range(generations):
    #     generation = Evol.recombinate(offsize)
    #     index_list = query_fitness_for_pops(generation)
    #     generation_acc = []
    #     for indi in generation.pops:
    #         generation_acc.append(indi.mean_acc)
    #     print(max(generation_acc))
    #     select_dict['gen'] = index_list
    #     Evol.environmental_selection(gen, generation, elitism)
    #     arg_index = Evol.pops.get_sorted_index_order_by_acc()
    #     best = Evol.pops.get_individual(arg_index[0])
    #     model_spec = ModelSpec(matrix=best.indi['matrix'], ops=best.indi['op_list'])
    #     index = nasbench101.get_index_by_model_spec(model_spec)
    #     best_list.append(index)
        
    #     info = '{}th generation: '.format(gen + 1) + 'optimal architecture: {}'.format(best.mean_acc)
    #     print(info)


    # for gen in tqdm(range(generations), desc='Searching for the Optimal Architecture'):
    for gen in range(generations):
        arg_index = Evol.pops.get_sorted_index_order_by_acc()
        best = Evol.pops.get_individual(arg_index[0])
        print(best.mean_acc)
        generation = Evol.recombinate(offsize)
        query_fitness_for_pops(generation)
        generation_acc = []
        for indi in generation.pops:
            generation_acc.append(indi.mean_acc)
        print(max(generation_acc))
        offspring = contrast_select(RelativeNet, Evol.pops.pops, generation, evalnum, device)
        index_list = query_fitness_for_pops(offspring)
        offspring_acc = []
        for indi in offspring.pops:
            offspring_acc.append(indi.mean_acc)
        print(max(offspring_acc))
        select_dict['gen'] = index_list
        Evol.environmental_selection(gen, offspring, elitism)
        arg_index = Evol.pops.get_sorted_index_order_by_acc()
        best = Evol.pops.get_individual(arg_index[0])
        model_spec = ModelSpec(matrix=best.indi['matrix'], ops=best.indi['op_list'])
        index = nasbench101.get_index_by_model_spec(model_spec)
        best_list.append(index)
        
        # data_sampled = sample_expansion(data_sampled, offspring)
        
        # sample = len(data_sampled)
        # if sample == 150 or sample == 200 or sample == 250 :# model adaptive update
        #     print('Updating Model...')
        #     dataset = TripletDataset(data_sampled=data_sampled, threshold=threshold, augmentation=True)
        #     train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        #     TripletNet = refine_TripletNetwork(TripletNet, epoch, train_loader, optimizer, criterion, device)
        # if sample == 150:
        #     save_path = os.path.join(current_path, 'predictor', 'NASBench101', 'TripletNet-200.pt')
        #     TripletNet.load_state_dict(torch.load(save_path))
        # elif sample == 200:
        #     save_path = os.path.join(current_path, 'predictor', 'NASBench101', 'TripletNet-300.pt')
        #     TripletNet.load_state_dict(torch.load(save_path))
        # elif sample == 250:
        #     save_path = os.path.join(current_path, 'predictor', 'NASBench101', 'TripletNet-400.pt')
        #     TripletNet.load_state_dict(torch.load(save_path))
        
        info = '{}th generation: '.format(gen + 1) + 'optimal architecture: {}'.format(best.mean_acc)
        print(info)

    save_path = os.path.join(current_path, 'pkl', 'evolve', 'select_dict.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(select_dict, file)
    save_path = os.path.join(current_path, 'pkl', 'evolve', 'best_list.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(best_list, file)

    valid = []
    test = []
    for index in best_list:
        fixed_stat, computed_stat = nasbench101.get_info_by_index(index)
        final_test_accuracy_list = []
        final_valid_accuracy_list = []
        for i in range(3):
            final_valid_accuracy_list.append(computed_stat[108][i]['final_validation_accuracy'])
            final_test_accuracy_list.append(computed_stat[108][i]['final_test_accuracy'])
        mean_final_valid_accuracy = np.mean(final_valid_accuracy_list)
        mean_final_test_accuracy = np.mean(final_test_accuracy_list)

        valid.append(mean_final_valid_accuracy)
        test.append(mean_final_test_accuracy)
    print('Search Path: ')
    print(valid)
    print(test)

    arg_index = Evol.pops.get_sorted_index_order_by_acc()
    best_individual = Evol.pops.get_individual(arg_index[0])
    print('Global optimal solution:')
    print(best_individual)
    model_spec = ModelSpec(matrix=best_individual.indi['matrix'], ops=best_individual.indi['op_list'])
    print(nasbench101.get_info_by_model_spec(model_spec))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity-Based Search on NASBench101')
    parser.add_argument('--pc', type=float, default=0.8, 
                        help='Crossover Probability')
    parser.add_argument('--pm', type=float, default=0.5, 
                        help='Mutation Probability')
    parser.add_argument('--m_num_matrix', type=int, default=1,
                        help='Number of Crossover Connection')
    parser.add_argument('--m_num_operation', type=int, default=1,
                        help='Number of Crossover Operation')
    parser.add_argument('--popsize', type=int, default=50,
                        help='Population Size')
    parser.add_argument('--offsize', type=int, default=200,
                        help='Offspring Size')
    parser.add_argument('--elitism', type=float, default=0.5,
                        help='Elite Rate')
    parser.add_argument('--gens', type=int, default=20,
                        help='Generations')
    parser.add_argument('--evalnum', type=int, default=20,
                        help='Number of Individuals Evaluated Per Generation')
    parser.add_argument('--number', type=int, default=100, 
                        help='Number of Samples')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Number of Refining Epochs of Model')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch Size')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Similarity Threshold')
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA Device Number')
    args = parser.parse_args()
    main(args)
import numpy as np
from enum import Enum
import fitness_function as ff

def initialize_population(pop_size, problem_size):
    population = np.random.randint(low=32, high=127, size=(pop_size, problem_size))
    return population


def evaluate(pop, f_func):
    return np.array(list(map(f_func, pop)))


def variate(pop, crossover_mode):
    (num_inds, num_params) = np.shape(pop)
    indices = np.array(range(num_inds))

    offsprings = []
    np.random.shuffle(indices)

    for i in range(0, num_inds, 2):
        index1 = indices[i]
        index2 = indices[i+1]
        offspring1 = pop[index1].copy()
        offspring2 = pop[index2].copy()

        if crossover_mode == Crossover.ONEPOINT:
            point = np.random.randint(low=0, high=num_params-1)
            offspring1[:point], offspring2[:point] = offspring2[:point], offspring1[:point].copy()
        else:
            for j in range(num_params):
                if np.random.randint(low=0, high=2) == 1:
                    offspring1[j], offspring2[j] = offspring2[j], offspring1[j]

        offsprings.append(offspring1)
        offsprings.append(offspring2)

    return np.reshape(offsprings, (num_inds, num_params))

def tournament_selection(pool_fitness, tournament_size, selection_size):
    num_individuals = len(pool_fitness)
    indices = np.arange(num_individuals)
    selected_indices = []

    while len(selected_indices) < selection_size:
        np.random.shuffle(indices)

        for i in range(0, num_individuals, tournament_size):
            idx_tournament = indices[i:i+tournament_size]
            winner = list(filter(lambda x : pool_fitness[x] == max(pool_fitness[idx_tournament]), idx_tournament))
            selected_indices.append(np.random.choice(winner))

    return selected_indices

def POPOP(user_options, func_inf, seed_num=1):
    np.random.seed(seed_num)

    population = initialize_population(user_options.POP_SIZE, user_options.PROBLEM_SIZE)
    pop_fitness = evaluate(population, func_inf.F_FUNC)
    num_eval_func_calls = len(pop_fitness)

    selection_size = len(population)
    generation = 0

    while not converge(population):
        offsprings = variate(population, user_options.CROSSOVER_MODE)
        off_fitness = evaluate(offsprings, func_inf.F_FUNC)
        num_eval_func_calls += len(off_fitness)

        pool = np.vstack((population, offsprings))
        pool_fitness = np.hstack((pop_fitness, off_fitness))

        pool_indices = tournament_selection(pool_fitness, user_options.TOURNAMENT_SIZE, selection_size)
        population = pool[pool_indices]
        pop_fitness = pool_fitness[pool_indices]

        generation += 1
        best_sol = population[np.argmax(pop_fitness)]
        print('## Gen: {}'.format(generation))
        print('Best Solution: {}'.format(best_sol))
        print('Decode: {}'.format("".join([chr(param) for param in best_sol.flatten()])))


    print("".join([chr(param) for param in np.unique(population, axis=0).flatten()]))
    optimized_solution_found = user_options.PROBLEM_SIZE == np.unique(pop_fitness)
    print(optimized_solution_found[0], num_eval_func_calls)
    return (optimized_solution_found[0], num_eval_func_calls)

def converge(pop):
    return len(np.unique(pop, axis=0)) == 1

class Crossover(Enum):
    ONEPOINT = 1
    UX = 2

class POPOPConfig:
    NAME = 'POPOP'
    PROBLEM_SIZE = 4
    POP_SIZE = 4
    TOURNAMENT_SIZE = 4
    CROSSOVER_MODE = Crossover.UX

    def __init__(self, pop_size, problem_size=4, tournament_size=4, crossover_mode=Crossover.UX):
        self.PROBLEM_SIZE = problem_size
        self.POP_SIZE = pop_size
        self.TOURNAMENT_SIZE = tournament_size
        self.CROSSOVER_MODE = crossover_mode

ff.NAME = 'Do Minh Tu'
user_options = POPOPConfig(300, len(ff.NAME), 4, Crossover.UX)
func_inf = ff.FuncInf('Match String', ff.my_name)

POPOP(user_options, func_inf)
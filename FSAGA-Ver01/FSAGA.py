import numpy as np
import matplotlib.pyplot as plt
import time
import math

def F1(x):
    return np.sum(x**2)

def F2(x):
    A = 10
    return A * len(x) + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in x])

def F3(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def F4(x):
    return 1/4000 * np.sum(x**2) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def F5(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

def F6(x):
    sum = 0
    sum1 = 1
    for i in range(len(x)):
        sum += np.abs(x[i])
        sum1 *= np.abs(x[i])
    return sum + sum1

def F7(x):
    m = 10
    return -np.sum(np.sin(x) * np.sin(((np.arange(1, len(x) + 1) * x**2) / np.pi)**(2 * m)))

def F8(x):
    sum = 0
    for i in range(len(x)):
        sum += (x[i] + 0.5)**2
    return sum

def F9(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.exp(1)

def flamingo_search(objective_function, initial_solution, n_iter, n_flamingos, sigma):
    best_solution = initial_solution
    best_objective_value = objective_function(best_solution)
    convergence = [best_objective_value]

    for i in range(n_iter):
        flamingos = [best_solution + np.random.normal(scale=sigma, size=len(best_solution)) for _ in range(n_flamingos)]
        for f in flamingos:
            f_value = objective_function(f)
            if f_value < best_objective_value:
                best_solution = f
                best_objective_value = f_value
        if i % 50 == 0:
            convergence.append(best_objective_value)

    convergence.append(best_objective_value)
    return best_solution, convergence

def create_population(size, dim, lb=-5, ub=10):
    return np.random.uniform(lb, ub, (size, dim))

def fitness_function(population, objective_function):
    fitness = np.array([objective_function(ind) for ind in population])
    fitness_prob = 1 / (fitness + 1e-10)
    fitness_prob -= np.min(fitness_prob)
    fitness_prob /= np.sum(fitness_prob)
    return fitness_prob

def selection(population, fitness, num_parents):
    parents_idx = np.random.choice(len(population), size=num_parents, replace=False, p=fitness)
    return population[parents_idx]

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        crossover_point = np.random.randint(1, offspring_size[1] - 1)
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, mutation_rate=0.1):
    for idx in range(offspring_crossover.shape[0]):
        if np.random.rand() < mutation_rate:
            random_index = np.random.randint(0, offspring_crossover.shape[1])
            offspring_crossover[idx, random_index] += np.random.normal()
    return offspring_crossover

def genetic_algorithm(population, objective_function, generations, mutation_rate=0.1, elitism_rate=0.1):
    convergence = []
    best_objective_value = np.inf

    for gen in range(generations):
        fitness = fitness_function(population, objective_function)
        parents = selection(population, fitness, len(population) // 2)
        offspring_crossover = crossover(parents, (len(population) - len(parents), population.shape[1]))
        offspring_mutation = mutation(offspring_crossover, mutation_rate)
        population[len(parents):] = offspring_mutation

        num_elites = int(elitism_rate * len(population))
        elites = population[np.argsort([objective_function(ind) for ind in population])[:num_elites]]
        population[:num_elites] = elites

        best_solution = population[np.argmin([objective_function(ind) for ind in population])]
        current_best_value = objective_function(best_solution)
        if current_best_value < best_objective_value:
            best_objective_value = current_best_value
        if gen % 50 == 0:
            convergence.append(best_objective_value)

    convergence.append(best_objective_value)
    return best_solution, convergence

def flamingo_genetic_search(objective_function, initial_solution, n_iter, n_flamingos, sigma, generations, population_size):
    fs_best_solutions = [flamingo_search(objective_function, initial_solution, n_iter, n_flamingos, sigma)[0] for _ in range(10)]
    population = create_population(population_size, len(initial_solution))
    population[:10] = fs_best_solutions
    best_solution, convergence = genetic_algorithm(population, objective_function, generations)
    return best_solution, convergence

def format_results(fs_convergence, ga_convergence, fsga_convergence):
    results = []
    for i in [100, 200, 300]:
        fs_fitness = fs_convergence[i // 50]
        ga_fitness = ga_convergence[i // 50]
        fsga_fitness = fsga_convergence[i // 50]
        results.append(f"Iteration {i}\tFSA: {fs_fitness:.4e}\tGA: {ga_fitness:.4e}\tFSA-GA: {fsga_fitness:.4e}")
    return results

def run_experiment(dim):
    functions = [F1, F2, F3, F4, F5, F6, F7, F8, F9]
    function_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7" , "F8", "F9"]

    n_iter = 300
    n_flamingos = 100
    sigma = 0.5
    generations = 300
    population_size = 100

    fig, axs = plt.subplots(3, 3, figsize=(12, 8))
    axs = axs.ravel()

    for i, (func, func_name) in enumerate(zip(functions, function_names)):
        initial_population = create_population(10, dim)
        initial_objective_values = np.array([func(ind) for ind in initial_population])
        best_initial_index = np.argmin(initial_objective_values)
        initial_solution = initial_population[best_initial_index]
        initial_value = initial_objective_values[best_initial_index]

        time_start = time.time()

        fs_best_solution, fs_convergence = flamingo_search(func, initial_solution, n_iter, n_flamingos, sigma)
        fs_convergence.insert(0, initial_value)

        population = create_population(population_size, dim)
        population[0] = initial_solution
        ga_best_solution, ga_convergence = genetic_algorithm(population, func, generations)
        ga_convergence.insert(0, initial_value)

        fsga_best_solution, fsga_convergence = flamingo_genetic_search(func, initial_solution, n_iter, n_flamingos, sigma, generations, population_size)
        fsga_convergence.insert(0, initial_value)

        time_end = time.time()

        x_points = np.arange(0, len(fs_convergence) * 50, 50)
        convergence_arrays = [fs_convergence, ga_convergence, fsga_convergence]
        min_length = min(map(len, convergence_arrays))
        convergence_arrays = [arr[:min_length] for arr in convergence_arrays]
        x_points = x_points[:min_length]

        axs[i].plot(x_points, convergence_arrays[0], label='FSA', linestyle='--')
        axs[i].plot(x_points, convergence_arrays[1], label='GA', linestyle='-.')
        axs[i].plot(x_points, convergence_arrays[2], label='FSA-GA')
        axs[i].set_title(func_name)
        axs[i].set_xlabel('Iterations')
        axs[i].set_ylabel('Fitness')    
        if func_name != "F7":
            axs[i].set_yscale('log')  
        axs[i].legend()
        axs[i].set_xlim(0, 300)

    plt.tight_layout()
    plt.show()

for dim in [10, 30, 50]:
    print(f"Kết quả của không gian dim = {dim}")
    run_experiment(dim)

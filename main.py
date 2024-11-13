import pygad, numpy

# Dados do problema
function_inputs = list({103,209,123,342,567,198,415,876,145,320,255,689,334,134,562,435,778,156,901,324})
desired_output = 3953

# Função de fitness adaptada para o Subset Sum Problem
def fitness_func(ga_instance, solution, solution_idx):
    # Multiplica cada gene pela entrada correspondente para formar o subconjunto
    subset_sum = numpy.sum(numpy.array(function_inputs) * solution)
    
    # A fitness é inversamente proporcional à diferença da soma desejada
    if subset_sum == desired_output:
        # Caso a soma seja exatamente o valor desejado, dá um fitness alto
        fitness = 1_000_000
    else:
        fitness = 1.0 / (1.0 + abs(subset_sum - desired_output))

    return fitness

# Parâmetros do algoritmo genético
fitness_function = fitness_func
num_generations = 1000
num_parents_mating = 4
sol_per_pop = 10              # soluções por população
num_genes = len(function_inputs)

# Inicializa os genes como binários
init_range_low = 0
init_range_high = 1

parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

# Criação do algoritmo genético
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    gene_space=[0, 1],  # Restringe os valores dos genes a binários (0 e 1)
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
    suppress_warnings=True
)

# Executa o algoritmo genético
ga_instance.run()
ga_instance.plot_fitness()

# Imprime a melhor solução encontrada
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parâmetros da melhor solução : {solution}".format(solution=solution))
print("Valor da fitness da melhor solução = {solution_fitness}".format(solution_fitness=solution_fitness))

# Calcula a soma com base na melhor solução encontrada
subset_sum = numpy.sum(numpy.array(function_inputs) * solution)
print("Soma prevista com base na melhor solução : {subset_sum}".format(subset_sum=subset_sum))

# Verifica quais elementos foram selecionados
selected_elements = [function_inputs[i] for i in range(len(solution)) if solution[i] == 1]
print("Elementos selecionados para o Subset Sum:", selected_elements)
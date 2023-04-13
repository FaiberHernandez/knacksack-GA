from random import *
import math
from os import system
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import csv

class Item(object):
    def __init__(self, w, v):
        self.weight = w
        self.value = v

"""
    Parametros que variaran en las pruebas
"""
POP_SIZE = 1300 #tamanho de la poblacion -20 da 999
GEN_MAX = 350 #Numero de generaciones
parent_eligibility = 1
mutation_chance = 0.87
parent_lottery = 0.05
crossover_rate = 0.9
elite_percentage = 0.1

ITEMS = []
CAPACITY = 0

#Reading csv file
with open('input1.csv', newline='') as csvfile:
  reader = csv.DictReader(csvfile, delimiter=";")
  line_count = 0
  for row in reader:
    if line_count == 0:
      CAPACITY = int(row['capacidad_maxima'])
    peso = int(row['peso'])
    valor = int(row['valor'])
    ITEMS.append(Item(peso, valor))
    line_count += 1

def fitness(spawn_vector):
    weight = fitness_weight(spawn_vector)
    value = sum(spawn_vector[i]*ITEMS[i].value for i in range(len(spawn_vector)))
    return value if weight <= CAPACITY else 0

def spawn_starting_population(amount):
    return [spawn_individual() for x in range(amount)]

def fitness_weight(spawn_vector):
    return sum(spawn_vector[i]*ITEMS[i].weight for i in range(len(spawn_vector)))

def check_individual_weight(spawn_vector):
    return fitness_weight(spawn_vector) <= CAPACITY

def spawn_individual():
    spawn_vector = [randint(0,1) for x in range(len(ITEMS))]

    #Check the fitness weight
    while check_individual_weight(spawn_vector) == False:
        spawn_vector.remove(1)
        spawn_vector.append(0)
    return spawn_vector

def mutate(target):
    """
    Changes a random element of the permutation array from 0 -> 1 or from 1 -> 0.
    """ 
    r = randint(0,len(target)-1)
    if target[r] == 1:
        target[r] = 0
    else:
        target[r] = 1

def evolve_population(pop):
    parent_length = int(parent_eligibility*len(pop))
    elite_length = int(elite_percentage*len(pop))
    parents = pop[:parent_length]
    nonparents = pop[parent_length:]

    # Elite strategy
    elites = sorted(parents, key=lambda x: fitness(x), reverse=True)[:elite_length]
    parents = elites + nonparents

    # Parent lottery!
    for np in nonparents:
        if parent_lottery > random():
            parents.append(np)

    # Mutation rate
    for p in parents:
        if mutation_chance > random():
            mutate(p)

    # Breeding! Close the doors, please.
    children = []
    desired_length = len(pop) - len(parents)
    while len(children) < desired_length :
        male = pop[randint(0,len(parents)-1)]
        female = pop[randint(0,len(parents)-1)]
        
        if random() < crossover_rate:
            half = math.floor(len(male)/2)
            child = male[:half] + female[half:] # from start to half from father, from half to end from mother
        else:
            child = male
            
        if mutation_chance > random():
            mutate(child)
        children.append(child)

    parents.extend(children)
    return parents

#Inicializacion y ejecucion del problema
system("cls")
list_best_exec = []
list_endtime_exec = []
list_iter_exec = []
excution = 1

for e in range(0, 30):
    list_best_values = []
    generation = 1
    best_solution = None
    best_value = None
    best_weight = None
    start_time = timeit.default_timer() 
    population = spawn_starting_population(POP_SIZE)
    for g in range(0,GEN_MAX):
        for i in population:
            o = fitness(i)

            if best_solution is None:
                best_solution = i
                best_value = o
                best_weight = fitness_weight(i) 
            elif o > best_value:
                best_solution = i
                best_value = o
                best_weight = fitness_weight(i)   
        list_best_values.append(best_value)
        population = evolve_population(population)
        #print("Generacion "+str(generation)+"/"+str(GEN_MAX)+" procesada!")
        generation += 1
    
    stop_time = timeit.default_timer()
    end_time = stop_time - start_time
    list_endtime_exec.append(end_time)
    list_best_exec.append(best_value)
    ngen_best = list_best_values.index(best_value) + 1
    list_iter_exec.append(ngen_best)
    print("Ejecución #", str(excution))
    print("MEJOR SOLUCION - iteración ", ngen_best," :",best_solution," peso: ", best_weight, " valor: ",best_value)
    print("Tiempo de ejecución", str(end_time))
    excution += 1
plt.plot(list_best_values)
plt.xlabel('Generación')
plt.ylabel('Mejor valor')
plt.title('Evolución del mejor valor por generación - ejecución '+str(excution - 1))
plt.show()

plt.plot(list_best_exec)
plt.xlabel('Ejecucion')
plt.ylabel('Mejor valor')
plt.title('Evolución del mejor valor por ejecucion')
plt.show()

plt.plot(list_endtime_exec)
plt.xlabel('Ejecucion')
plt.ylabel('Tiempo de ejecución')
plt.title('Evolución del tiempo de ejecución por corrida')
plt.show()

df = pd.DataFrame(list_best_exec, columns =['Fitness'], dtype = int)
df['Time'] = list_endtime_exec
df['Iteracion'] = list_iter_exec

print(df.head())
df.to_excel('AG4.xlsx')


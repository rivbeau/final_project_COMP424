import random
import numpy as np
import json
from agents.riv_agent import RivAgent
from agents.riv_agent_opp import RivAgentOpp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from world import World
from simulator import Simulator


#NEED to fix the fitness so its not like so fucking long 



# Features used:
# 1. piece difference (material advantage)
# 2. edge control (occupying edge squares)
# 3. adj block (control of squares adjacent to opponent's pieces)
# 4. centrality bonus (control of central squares)
# 5. danger risk (low vs big advantage) -> to be included or excluded ? 


type Individual = List[List[float]]  #5 weights

def random_individual() -> Individual:
    weights = [random.uniform(-1, 1) for _ in range(5)]
    temp = [random.uniform(0.5, 2.0) for _ in range(5)]
    return [weights, temp]
    

def init_population(N: int) -> List[Individual]:
    return [random_individual() for _ in range(N)]
    
            
def write_weights(weights, idx, suffix): # only works with "" and "_opp"
    name = f"riv_weights_{idx}_{suffix}.json"
    with open (name, "w") as f:
        json.dump(weights, f)
        
    return name
    
def run_game(idx, ind1, ind2, N):
    w1, t1 = ind1
    w2, t2 = ind2
    
    
    sim = Simulator(
        player_1="riv_agent",
        player_2="riv_agent",
        weights1=w1,
        temp1=t1,
        weights2=w2,
        temp2=t2,
        autoplay=False,
        display=False,
        display_delay=2, 
        autoplay_runs=N,
    )
    
    win_p1 = sim.run_autoplay()
    print(win_p1)
    return idx, win_p1
    # f1_path = write_weights(w1,idx, "")
    # f2_path = write_weights(w2,idx, "opp")
    
    # pattern = r"Player 1, agent .* win percentage: (\d+(\.\d+)?)"
    # match = re.search(pattern, output)
    # if match:
    #     print(f"good {match.group(1)}")
    #     return idx, float(match.group(1)) * 5.0 
    # else:
    #     print("Could not parse output:")
    #     print(output)
    #     return idx, 0.0


def fitness_parallel(pop: List[Individual], opponents_per_ind=3) -> List[float]:
    fitness = [0.0] * len(pop)
    jobs = []
    with ProcessPoolExecutor() as executor:
        for i, ind in enumerate(pop):
            opponents = random.sample(pop, opponents_per_ind)
            for opp in opponents:
                jobs.append(executor.submit(run_game, i, ind, opp, 5))
                
            jobs.append(executor.submit(run_game, i, ind, [[1,0,0,0,0], [1,1,1,1,1]], 10))
            
        for future in as_completed(jobs):
            idx, result = future.result()
            fitness[idx] += result
    
    return fitness
    
    
    # for weights in pop: #for all ind in pop
    #     wins = 0
    #     for opp in random.sample(pop, 3): # 3 random opponents from gen
    #         wins += run_game(weights, opp)
    #     wins += 2 * run_game(weights, [1,0,0,0,0])
    #     fitness.append(wins)
    # return fitness

def tournament_select(pop, fits, k=3):
    n = len(pop)
    cand = random.sample(list(range(n)), k)
    best = max(cand, key=lambda idx: fits[idx])
    return best

def reproduce(p1, p2) -> Individual:
    w1, t1 = p1
    w2, t2 = p2
    point = random.randint(0, 4)
    
    child_weights = w1[:point] + w2[point:]
    child_temps = t1[:point] + t2[point:]
    child = [child_weights, child_temps]
    return child

def mutate_pop(pop, mutation_rate=0.10, mutation_strength=0.4):
    new_pop = []
    for weights, temps in pop:
        new_w = []
        for w in weights:
            if random.random() < mutation_rate:
                w += random.gauss(0, mutation_strength)
                # w = max(-1, min(1, w)) hard limit [-1, 1]
            new_w.append(w)
        new_t = []
        for t in temps:
            if random.random() < mutation_rate:
                t += random.gauss(0, mutation_strength/2)
                # t = max(0.1, min(5, t)) hard limit [0.1, 5]
            new_t.append(t)
        new_pop.append([new_w, new_t])
    return new_pop



def simulate(G: int, N: int) -> List[Individual]:
    pop = init_population(N)
    for generation in range(G):
        print(f"GEN {generation} working")
        
        fits = fitness_parallel(pop)
        best_pop = max(range(len(fits)), key=lambda i: fits[i])
        print(f"best of the generation: {pop[best_pop]}")
        new_pop = [pop[best_pop]]# Elitism: carry best individual to next gen
        for _ in range(N-1):
            p1_idx = tournament_select(pop, fits)
            p2_idx = tournament_select(pop, fits)
            child = reproduce(pop[p1_idx], pop[p2_idx])
            new_pop.append(child)
        rate = max(0.05, 0.2 * (1 - generation/G))
        strength = 0.4
        pop = mutate_pop(new_pop, mutation_rate=rate, mutation_strength=strength)
    return pop


def main():
    best_pop = simulate(20, 20)
    print("it finished")
    print(best_pop)


if __name__ == "__main__":
    main()
    

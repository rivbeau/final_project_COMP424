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


type Individual = List[float]  #5 weights

def random_individual() -> Individual:
    return [random.uniform(-1, 1) for _ in range(5)]
    

def init_population(N: int) -> List[Individual]:
    return [random_individual() for _ in range(N)]
    
            
def write_weights(weights, idx, suffix): # only works with "" and "_opp"
    name = f"riv_weights_{idx}_{suffix}.json"
    with open (name, "w") as f:
        json.dump(weights, f)
        
    return name
    
def run_game(idx,w1, w2):
    
    sim = Simulator(
        player_1="riv_agent",
        player_2="riv_agent",
        weights1=w1,
        weights2=w2,
        autoplay=False,
        display=False,
        display_delay=2,
        autoplay_runs=5,
    )
    
    win_p1 = sim.run_autoplay()
        
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
        for i, weights in enumerate(pop):
            opponents = random.sample(pop, opponents_per_ind)
            for opp in opponents:
                jobs.append(executor.submit(run_game, i, weights, opp))
                
            jobs.append(executor.submit(run_game,i, weights, [1,0,0,0,0]))
            
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

def tournament_select(pop, fits, k=4):
    cand = random.sample(list(range(20)), k)
    best = max(cand, key=lambda idx: fits[idx])
    return best

def reproduce(p1, p2) -> Individual:
    point = random.randint(0, 4)
    
    child = p1[:point] + p2[point:]
    return child

def mutate_pop(pop, mutation_rate=0.20, mutation_strength=0.4):
    new_pop = []
    for ind in pop:
        new_ind = []
        for w in ind:
            if random.random() < mutation_rate:
                w += random.gauss(0, mutation_strength)
                w = max(-1, min(1, w))
            new_ind.append(w)
        new_pop.append(new_ind)
    return new_pop



def simulate(G: int, N: int) -> List[Individual]:
    pop = init_population(N)
    for generation in range(G):
        print(f"GEN {generation} working")
        
        fits = fitness_parallel(pop)
        best_pop = max(range(len(fits)), key=lambda i: fits[i])
        print(f"best of the generation: {pop[best_pop]}")
        new_pop = []
        for _ in range(N):
            p1_idx = tournament_select(pop, fits)
            p2_idx = tournament_select(pop, fits)
            child = reproduce(pop[p1_idx], pop[p2_idx])
            new_pop.append(child)
            
        pop = mutate_pop(new_pop)
    return pop


def main():
    best_pop = simulate(20, 20)
    print("it finished")
    print("best ind", best_pop[0])


if __name__ == "__main__":
    main()
    

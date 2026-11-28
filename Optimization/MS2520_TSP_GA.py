
import random
import math
import matplotlib.pyplot as plt


def load_cities(filename):
    cities = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
            name = " ".join(parts[:-2])
            x, y = float(parts[-2]), float(parts[-1])
            cities.append((name, x, y))
    if len(cities) < 3:
        raise ValueError("Need at least 3 cities in the dataset.")
    return cities



def distance(city_a, city_b):
    return math.hypot(city_a[1] - city_b[1], city_a[2] - city_b[2])


def route_distance(route, cities):
    total = 0.0
    for i in range(len(route)):
        total += distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return total


def route_fitness(route, cities):
    return 1.0 / route_distance(route, cities)



def create_route(n):
    route = list(range(n))
    random.shuffle(route)
    return route


def initial_population(pop_size, n):
    return [create_route(n) for _ in range(pop_size)]


def rank_population(population, cities):
    """Return list of (index_in_population, fitness), best first."""
    scored = [(i, route_fitness(r, cities)) for i, r in enumerate(population)]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def selection(ranked, elite_size):
    """Elitism + fitness-proportionate (roulette wheel) selection."""
    selected = [idx for idx, _ in ranked[:elite_size]]
    fitness_sum = sum(f for _, f in ranked)
    for _ in range(len(ranked) - elite_size):
        pick = random.uniform(0, fitness_sum)
        running = 0.0
        for idx, f in ranked:
            running += f
            if running >= pick:
                selected.append(idx)
                break
    return selected


def mating_pool(population, selected_indices):
    return [population[i] for i in selected_indices]


def ordered_crossover(parent1, parent2):
    """Order Crossover (OX) -- keeps a slice of parent1, fills the rest
    in parent2's order, so every city appears exactly once."""
    a, b = sorted(random.sample(range(len(parent1)), 2))
    slice1 = parent1[a:b]
    rest = [gene for gene in parent2 if gene not in slice1]
    return slice1 + rest


def breed_population(pool, elite_size):
    children = pool[:elite_size]  # carry the elites over unchanged
    shuffled = random.sample(pool, len(pool))
    for i in range(len(pool) - elite_size):
        child = ordered_crossover(shuffled[i], shuffled[len(pool) - i - 1])
        children.append(child)
    return children


def mutate(route, mutation_rate):
    """Swap mutation: randomly swap pairs of cities."""
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route


def mutate_population(population, mutation_rate):
    return [mutate(route, mutation_rate) for route in population]



def plot_route(route, cities, generation, best_distance):
    xs = [cities[i][1] for i in route] + [cities[route[0]][1]]
    ys = [cities[i][2] for i in route] + [cities[route[0]][2]]
    names = [cities[i][0] for i in route]

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, "o-", color="steelblue")
    for i, name in enumerate(names):
        plt.annotate(name, (xs[i], ys[i]), fontsize=8, xytext=(4, 4),
                     textcoords="offset points")
    plt.title(f"Best route at generation {generation}  (distance = {best_distance:.2f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


def plot_fitness_history(avg_fitness_per_gen):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(avg_fitness_per_gen) + 1), avg_fitness_per_gen, color="darkorange")
    plt.title("Average population fitness vs. generation")
    plt.xlabel("Generation")
    plt.ylabel("Average fitness (1 / distance)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def run_genetic_algorithm(filename, pop_size=100, elite_size=20,
                           mutation_rate=0.01, generations=300,
                           plot_every=50):
    cities = load_cities(filename)
    n = len(cities)
    population = initial_population(pop_size, n)
    avg_fitness_per_gen = []

    for gen in range(1, generations + 1):
        ranked = rank_population(population, cities)
        avg_fit = sum(f for _, f in ranked) / len(ranked)
        avg_fitness_per_gen.append(avg_fit)

        best_route = population[ranked[0][0]]
        best_dist = route_distance(best_route, cities)

        if gen == 1 or gen % plot_every == 0 or gen == generations:
            print(f"Generation {gen:4d}  |  Best distance so far: {best_dist:.2f}")
            plot_route(best_route, cities, gen, best_dist)

        selected = selection(ranked, elite_size)
        pool = mating_pool(population, selected)
        children = breed_population(pool, elite_size)
        population = mutate_population(children, mutation_rate)

    # Final result
    ranked = rank_population(population, cities)
    best_route = population[ranked[0][0]]
    best_dist = route_distance(best_route, cities)
    route_names = [cities[i][0] for i in best_route]

    print("\nFinal best route:")
    print(" -> ".join(route_names + [route_names[0]]))
    print(f"Total distance: {best_dist:.2f}")

    plot_fitness_history(avg_fitness_per_gen)

    return best_route, best_dist, avg_fitness_per_gen



if __name__ == "__main__":
    DATA_FILE = "india_cities_GA.txt"   
    run_genetic_algorithm(
        filename=DATA_FILE,
        pop_size=100,
        elite_size=20,
        mutation_rate=0.01,
        generations=300,
        plot_every=50,
    )
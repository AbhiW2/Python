
import os
import random
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE    = "India_cities_GA.txt"
N_ANTS       = 30        # ants per iteration
N_ITER       = 100       # total iterations
ALPHA        = 1.0       # pheromone importance
BETA         = 5.0       # heuristic (1/distance) importance
RHO          = 0.5       # evaporation rate
Q            = 100.0     # pheromone deposit constant
PLOT_EVERY   = 20        # draw path at every PLOT_EVERY iterations
SEED         = 42
OUT_DIR      = "aco_plots"

random.seed(SEED)
np.random.seed(SEED)


def load_cities(filename):
    names, coords = [], []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace("\t", ",").split(",")] \
                if "," in line else line.split()
            if len(parts) < 3:
                continue
            try:
                lat, lon = float(parts[-2]), float(parts[-1])
            except ValueError:
                continue                       # header line etc.
            names.append(" ".join(parts[:-2]))
            coords.append((lat, lon))
    return names, np.array(coords)


def haversine_matrix(coords):
    """Great-circle distance (km) between every pair of cities."""
    R = 6371.0
    lat = np.radians(coords[:, 0])[:, None]
    lon = np.radians(coords[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def tour_length(tour, dist):
    return sum(dist[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def plot_path(tour, names, coords, title, filename, show=True):
    fig, ax = plt.subplots(figsize=(8, 9))
    xs = [coords[c, 1] for c in tour] + [coords[tour[0], 1]]   # longitude
    ys = [coords[c, 0] for c in tour] + [coords[tour[0], 0]]   # latitude
    ax.plot(xs, ys, "-o", color="tab:blue", lw=1.8, ms=7, mfc="red", mec="black")

    # arrows to show direction
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        ax.annotate("", xy=(coords[b, 1], coords[b, 0]),
                    xytext=(coords[a, 1], coords[a, 0]),
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))

    # city names + visiting order
    for order, c in enumerate(tour, 1):
        ax.annotate(f"{order}. {names[c]}", (coords[c, 1], coords[c, 0]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.scatter(coords[tour[0], 1], coords[tour[0], 0], s=180, c="lime",
               edgecolors="black", zorder=5, label=f"Start: {names[tour[0]]}")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def aco_tsp(dist, names, coords):
    n = len(dist)
    eta = 1.0 / (dist + np.eye(n))        # heuristic visibility (avoid /0)
    np.fill_diagonal(eta, 0)
    tau = np.ones((n, n))                 # initial pheromone

    best_tour, best_len = None, float("inf")
    history = []

    os.makedirs(OUT_DIR, exist_ok=True)

    for it in range(1, N_ITER + 1):
        all_tours, all_lens = [], []

        for _ in range(N_ANTS):
            start = random.randrange(n)
            tour, visited = [start], {start}
            while len(tour) < n:
                cur = tour[-1]
                cand = [j for j in range(n) if j not in visited]
                w = np.array([(tau[cur, j] ** ALPHA) * (eta[cur, j] ** BETA)
                              for j in cand])
                p = w / w.sum()
                nxt = cand[np.random.choice(len(cand), p=p)]
                tour.append(nxt)
                visited.add(nxt)
            L = tour_length(tour, dist)
            all_tours.append(tour)
            all_lens.append(L)
            if L < best_len:
                best_len, best_tour = L, tour[:]

        # pheromone evaporation + deposit
        tau *= (1 - RHO)
        for tour, L in zip(all_tours, all_lens):
            d = Q / L
            for i in range(n):
                a, b = tour[i], tour[(i + 1) % n]
                tau[a, b] += d
                tau[b, a] += d
        # elitist reinforcement of the best-so-far tour
        for i in range(n):
            a, b = best_tour[i], best_tour[(i + 1) % n]
            tau[a, b] += Q / best_len
            tau[b, a] += Q / best_len

        history.append(best_len)
        print(f"Iteration {it:3d} | iter best = {min(all_lens):9.2f} km "
              f"| global best = {best_len:9.2f} km")

        # path tracing at intermediate stages
        if it == 1 or it % PLOT_EVERY == 0:
            plot_path(best_tour, names, coords,
                      f"ACO - Iteration {it}  |  Best distance = {best_len:.2f} km",
                      os.path.join(OUT_DIR, f"path_iter_{it:03d}.png"))

    return best_tour, best_len, history


if __name__ == "__main__":
    names, coords = load_cities(DATA_FILE)
    print(f"Loaded {len(names)} cities from '{DATA_FILE}'")
    dist = haversine_matrix(coords)

    best_tour, best_len, history = aco_tsp(dist, names, coords)

    # rotate the tour so it starts from the first city of the file
    k = best_tour.index(0)
    best_tour = best_tour[k:] + best_tour[:k]

    print("\n===== FINAL RESULT =====")
    print(" -> ".join(names[c] for c in best_tour) + f" -> {names[best_tour[0]]}")
    print(f"Minimum total distance = {best_len:.2f} km")

    plot_path(best_tour, names, coords,
              f"ACO - FINAL ROUTE  |  Total distance = {best_len:.2f} km",
              os.path.join(OUT_DIR, "path_final.png"))

    # convergence curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(history) + 1), history, color="tab:green")
    plt.xlabel("Iteration")
    plt.ylabel("Best tour length (km)")
    plt.title("ACO convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "convergence.png"), dpi=130)
    plt.show()
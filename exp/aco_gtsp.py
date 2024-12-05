import numpy as np
import time

class ACO_GTSP:
    def __init__(self, distances, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5, iterations=100):
        self.distances = distances
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.num_cities = distances.shape[0]
        self.pheromones = np.ones(distances.shape)

    def _select_next_city(self, current_city, visited, cluster_info):
        probabilities = []
        for j in range(self.num_cities):
            if j not in visited and cluster_info[j] not in [cluster_info[v] for v in visited]:
                pheromone = self.pheromones[current_city, j] ** self.alpha
                visibility = (1 / self.distances[current_city, j]) ** self.beta
                probabilities.append(pheromone * visibility)
            else:
                probabilities.append(0)
        probabilities = np.array(probabilities)
        return np.random.choice(range(self.num_cities), p=probabilities / probabilities.sum())

    def _construct_solution(self, cluster_info):
        paths = []
        path_costs = []
        for ant in range(self.num_ants):
            path = []
            current_city = np.random.randint(0, self.num_cities)
            visited = [current_city]
            while len(set(cluster_info[v] for v in visited)) < len(set(cluster_info)):
                next_city = self._select_next_city(current_city, visited, cluster_info)
                visited.append(next_city)
                current_city = next_city
            visited.append(visited[0])  # Return to starting point
            paths.append(visited)
            path_costs.append(self._calculate_path_cost(visited))
        return paths, path_costs

    def _calculate_path_cost(self, path):
        return sum(self.distances[path[i], path[i + 1]] for i in range(len(path) - 1))

    def _update_pheromones(self, paths, path_costs):
        self.pheromones *= (1 - self.evaporation_rate)
        for path, cost in zip(paths, path_costs):
            for i in range(len(path) - 1):
                self.pheromones[path[i], path[i + 1]] += 1 / cost

    def run(self, cluster_info):
        best_path, best_cost = None, float('inf')
        start_time = time.time()
        for iteration in range(self.iterations):
            paths, path_costs = self._construct_solution(cluster_info)
            self._update_pheromones(paths, path_costs)
            iteration_best_cost = min(path_costs)
            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_path = paths[np.argmin(path_costs)]
        exec_time = time.time() - start_time
        return best_path, best_cost, exec_time, {'iterations': self.iterations}

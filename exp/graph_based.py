import heapq
import time

class GLPA:
    def __init__(self, distances):
        self.distances = distances
        self.num_cities = distances.shape[0]
        self.priority_queue = []  # Min-heap priority queue
        self.g_values = {i: float('inf') for i in range(self.num_cities)}
        self.rhs_values = {i: float('inf') for i in range(self.num_cities)}

    def _initialize(self, start_city):
        self.g_values[start_city] = float('inf')
        self.rhs_values[start_city] = 0
        heapq.heappush(self.priority_queue, (self._calculate_priority(start_city), start_city))

    def _calculate_priority(self, city):
        return min(self.g_values[city], self.rhs_values[city])

    def _update_state(self, city):
        if city != 0:  # Assume city 0 is the goal
            self.rhs_values[city] = min(
                self.g_values[neighbor] + self.distances[neighbor, city]
                for neighbor in range(self.num_cities)
                if self.distances[neighbor, city] != float('inf')
            )
        for i, (_, c) in enumerate(self.priority_queue):
            if c == city:
                self.priority_queue.pop(i)
                heapq.heapify(self.priority_queue)
                break
        if self.g_values[city] != self.rhs_values[city]:
            heapq.heappush(self.priority_queue, (self._calculate_priority(city), city))

    def run(self, start_city=0):
        start_time = time.time()
        self._initialize(start_city)
        visited = set()
        best_path = []
        best_cost = 0

        while self.priority_queue:
            _, current_city = heapq.heappop(self.priority_queue)
            if current_city in visited:
                continue
            visited.add(current_city)
            best_path.append(current_city)

            for neighbor in range(self.num_cities):
                if self.distances[current_city, neighbor] != float('inf') and neighbor not in visited:
                    self._update_state(neighbor)

            self.g_values[current_city] = self.rhs_values[current_city]

        best_path.append(start_city)
        best_cost = sum(self.distances[best_path[i], best_path[i + 1]] for i in range(len(best_path) - 1))
        exec_time = time.time() - start_time

        return best_path, best_cost, exec_time, {"iterations": len(visited)}

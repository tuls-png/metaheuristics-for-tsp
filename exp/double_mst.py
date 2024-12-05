import time
import networkx as nx

class DoubleMST:
    def __init__(self, distances):
        self.distances = distances
        self.num_cities = distances.shape[0]

    def _minimum_spanning_tree(self):
        G = nx.Graph()
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                G.add_edge(i, j, weight=self.distances[i, j])
        return nx.minimum_spanning_tree(G)

    def _eulerian_tour(self, mst):
        doubled_mst = nx.MultiGraph(mst)
        doubled_mst.add_edges_from(mst.edges)
        return list(nx.eulerian_circuit(doubled_mst))

    def _shortcut_tour(self, eulerian_tour):
        visited = []
        for u, v in eulerian_tour:
            if u not in visited:
                visited.append(u)
        visited.append(visited[0])
        return visited

    def run(self):
        start_time = time.time()
        mst = self._minimum_spanning_tree()
        eulerian_tour = self._eulerian_tour(mst)
        best_path = self._shortcut_tour(eulerian_tour)
        best_cost = sum(self.distances[best_path[i], best_path[i + 1]] for i in range(len(best_path) - 1))
        exec_time = time.time() - start_time
        return best_path, best_cost, exec_time, {'mst_weight': mst.size(weight='weight')}

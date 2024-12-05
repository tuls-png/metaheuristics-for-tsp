import numpy as np
import matplotlib.pyplot as plt
from aco_gtsp import ACO_GTSP
from double_mst import DoubleMST
from graph_based import GLPA

# Generate random test data
def generate_distances(num_cities):
    distances = np.random.randint(1, 100, size=(num_cities, num_cities)).astype(float)
    np.fill_diagonal(distances, np.inf)  # No self-loops
    return distances

def get_user_case(num_cities):
    """Categorize the problem based on the number of cities."""
    if num_cities <= 20:
        return "Small"
    elif num_cities <= 100:
        return "Medium"
    else:
        return "Large"

# Run all algorithms and compare
def run_comparison(num_cities):
    distances = generate_distances(num_cities)
    cluster_info = np.random.randint(0, num_cities // 2, size=num_cities)

    # Run ACO
    aco = ACO_GTSP(distances)
    aco_result = aco.run(cluster_info)

    # Run Double-MST
    double_mst = DoubleMST(distances)
    mst_result = double_mst.run()

    # Run Graph-Based Algorithm
    graph_based = GLPA(distances)
    graph_result = graph_based.run(start_city=0)

    # Compare Results
    results = {
        "ACO": {"path": aco_result[0], "cost": aco_result[1], "time": aco_result[2], "metrics": aco_result[3]},
        "Double-MST": {"path": mst_result[0], "cost": mst_result[1], "time": mst_result[2], "metrics": mst_result[3]},
        "Graph-Based": {"path": graph_result[0], "cost": graph_result[1], "time": graph_result[2], "metrics": graph_result[3]},
    }

    # Find the best algorithm for cost and time
    best_algo_cost = min(results, key=lambda x: results[x]['cost'])
    best_algo_time = min(results, key=lambda x: results[x]['time'])

    return distances, results, best_algo_cost, best_algo_time

# Function to plot the path
def plot_path(distances, path, title="Path Visualization"):
    plt.figure(figsize=(8, 6))
    city_coords = np.random.rand(len(distances), 2) * 100  # Random 2D coordinates for cities

    # Plot cities
    plt.scatter(city_coords[:, 0], city_coords[:, 1], color='red', zorder=5)
    for i, (x, y) in enumerate(city_coords):
        plt.text(x, y, str(i), fontsize=12, ha='right')

    # Plot the path
    path_coords = city_coords[path]
    plt.plot(path_coords[:, 0], path_coords[:, 1], marker='o', color='b', linestyle='-', zorder=4)

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# Main function to run comparisons and determine best algorithm
def main():
    # Test with different sizes of TSP instances
    for num_cities in [10, 50, 200]:  # Small, Medium, and Large cases
        print(f"\nRunning for {num_cities} cities ({get_user_case(num_cities)} case):")

        # Run comparison for the current number of cities
        distances, results, best_algo_cost, best_algo_time = run_comparison(num_cities)

        # Print results
        print(f"Results for {num_cities} cities:")
        for algo, result in results.items():
            print(f"{algo} Results: Best Path: {result['path']}, Cost: {result['cost']}, Time: {result['time']:.4f}s")

        # Visualize paths
        for algo, result in results.items():
            plot_path(distances, result['path'], title=f"{algo} Path Visualization")

        # Print the best algorithm based on cost and time
        print(f"\nBest Algorithm (Cost): {best_algo_cost} with Cost {results[best_algo_cost]['cost']}")
        print(f"Best Algorithm (Time): {best_algo_time} with Time {results[best_algo_time]['time']:.4f}s")

# Run the main function
if __name__ == "__main__":
    main()

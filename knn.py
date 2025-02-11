import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [24.412, 32.932], [35.19, 12.189], [26.288, 41.718], [0.376, 15.506], [26.116, 3.963],
    [25.893, 31.515], [23.606, 15.402], [28.026, 15.47], [26.36, 34.488], [23.013, 36.213],
    [27.819, 41.867], [39.634, 42.23], [35.477, 35.104], [25.768, 5.967], [-0.684, 21.105],
    [3.387, 17.81], [32.986, 3.412], [34.258, 9.931], [6.313, 29.426], [33.899, 37.535],
    [4.718, 12.125], [21.054, 5.067], [3.267, 21.911], [24.537, 38.822], [4.55, 16.179],
    [25.712, 7.409], [3.884, 28.616], [29.081, 34.539], [14.943, 23.263], [32.169, 45.421],
    [32.572, 16.944], [33.673, 13.163], [29.189, 13.226], [25.994, 34.444], [16.513, 23.396],
    [23.492, 11.142], [26.878, 36.609], [31.604, 36.911], [34.078, 33.827], [11.286, 16.082],
    [30.15, 9.642], [36.569, 45.827], [3.983, 11.839], [12.891, 23.832], [21.314, 13.264],
    [29.101, 44.781], [30.671, 9.294], [35.139, 9.803], [35.563, 42.759], [35.028, 15.779],
    [9.776, 16.988], [24.268, 5.693], [-0.36, 15.319], [33.062, 47.693], [21.034, 37.463],
    [31.806, 4.484], [22.493, 39.466], [29.056, 46.004], [29.822, 13.83], [35.439, 14.439]
])

data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def k_means(data, k, max_iters=100):
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return clusters, centroids

def plot_clusters(data, clusters, centroids, k):
    colors = ['r', 'g', 'b']
    for i in range(k):
        plt.scatter(data[clusters == i, 0], data[clusters == i, 1], c=colors[i], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'K-Means Clustering (k={k})')
    plt.legend()
    plt.show()

for k in [2, 3]:
    clusters, centroids = k_means(data, k)
    plot_clusters(data, clusters, centroids, k)

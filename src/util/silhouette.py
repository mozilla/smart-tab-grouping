import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances, silhouette_score

import numpy as np


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def average_distance(point, points):
    if len(points) == 0:
        return 0
    return np.mean([euclidean_distance(point, other) for other in points])


def silhouette_score_manual(X, labels):
    """
    Compute the average silhouette score for a clustering without using scikit-learn.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        Feature set.
    labels : array-like of shape (n_samples,)
        Cluster labels for each point in X.

    Returns:
    float
        The average silhouette score.
    """
    unique_labels = np.unique(labels)
    n_samples = X.shape[0]

    # Initialize a list to hold silhouette scores for each point
    silhouette_scores = []

    for i in range(n_samples):
        current_point = X[i]
        current_label = labels[i]

        # Get all points in the same cluster as the current point (excluding the point itself)
        same_cluster_points = X[(labels == current_label) & (np.arange(n_samples) != i)]

        # Intra-cluster distance (mean distance to other points in the same cluster)
        if len(same_cluster_points) > 0:
            a = average_distance(current_point, same_cluster_points)
        else:
            a = 0

        # Inter-cluster distance (mean distance to points in the nearest cluster)
        b = float('inf')
        for label in unique_labels:
            if label == current_label:
                continue
            other_cluster_points = X[labels == label]
            if len(other_cluster_points) > 0:
                distance_to_other_cluster = average_distance(current_point, other_cluster_points)
                b = min(b, distance_to_other_cluster)

        # Calculate silhouette score for the point
        if max(a, b) > 0:
            silhouette = (b - a) / max(a, b)
        else:
            silhouette = 0

        silhouette_scores.append(silhouette)

    # Return the mean silhouette score across all points
    return silhouette_scores, np.mean(silhouette_scores)


# Example usage:
# X = np.array([[1, 2], [3, 4], [5, 6], [8, 9], [10, 11]])
# labels = np.array([0, 0, 1, 1, 1])
# score = silhouette_score_manual(X, labels)
# print(f"Average silhouette score: {score}")


# Function to find the optimal number of clusters using silhouette score
def silh_find_optimal_k(X, cluster_space):
    best_k = None
    best_silhouette = -1
    silhouette_scores = []

    for k in cluster_space:
        print("k is ")
        print(k)
        if k >= X.shape[0]:
            break
        print(X.shape)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        # silhouette_avg = silhouette_score_manual(X, labels)
        # print(f"scikit {silhouette_scikit} custom {silhouette_avg}")
        silhouette_scores.append(silhouette_avg)

        print(f"Number of clusters: {k}, Silhouette Score: {silhouette_avg}")

        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_k = k
    print(f"Best k is {best_k}")
    return best_k

if __name__=="__main__":
    # Example dataset creation
    X, _ = make_blobs(n_samples=500, centers=4, random_state=42, cluster_std=1.0)

    # Define the range of K to test
    k_range = range(2, 11)

    # Find the optimal K based on silhouette score
    optimal_k, silhouette_scores = silh_find_optimal_k(X, k_range)

    print(f"The optimal number of clusters is: {optimal_k}")

    # Plot the silhouette scores for different K values
    import matplotlib.pyplot as plt

    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters, K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.show()

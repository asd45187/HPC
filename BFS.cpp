#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Structure to represent a graph
struct Graph {
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency list
};

// Function to add an edge to the graph
void addEdge(Graph& graph, int src, int dest) {
    graph.adj[src].push_back(dest);
}

// Parallel Breadth First Search
void parallelBFS(Graph& graph, int startVertex, vector<int>& distances) {
    // Mark all vertices as not visited
    vector<bool> visited(graph.V, false);

    // Create a queue for BFS
    queue<int> queue;

    // Mark the current node as visited and enqueue it
    visited[startVertex] = true;
    queue.push(startVertex);
    distances[startVertex] = 0;

    // Parallel BFS loop
    while (!queue.empty()) {
        // Dequeue a vertex from the queue
        int currentVertex = queue.front();
        queue.pop();

        // Parallelize the loop for visiting neighbors
        #pragma omp parallel for
        for (int i = 0; i < graph.adj[currentVertex].size(); ++i) {
            int neighbor = graph.adj[currentVertex][i];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                distances[neighbor] = distances[currentVertex] + 1;
                queue.push(neighbor);
            }
        }
    }
}

int main() {
    Graph graph;
    int numVertices = 6; // Example number of vertices
    graph.V = numVertices;
    graph.adj.resize(numVertices);

    // Example edges
    addEdge(graph, 0, 1);
    addEdge(graph, 0, 2);
    addEdge(graph, 1, 3);
    addEdge(graph, 1, 4);
    addEdge(graph, 2, 5);

    vector<int> distances(numVertices, -1); // Initialize distances to -1 (unreachable)

    // Start vertex for BFS
    int startVertex = 0;

    // Perform parallel BFS
    parallelBFS(graph, startVertex, distances);

    // Print distances from start vertex
    for (int i = 0; i < numVertices; ++i) {
        cout << "Distance from " << startVertex << " to " << i << " is " << distances[i] << endl;
    }

    return 0;
}

//output

// Distance from 0 to 0 is 0
// Distance from 0 to 1 is 1
//Distance from 0 to 2 is 1
//Distance from 0 to 3 is 2
//Distance from 0 to 4 is 2
//Distance from 0 to 5 is 2

//Explation of code

//Header Includes: The code includes necessary header files like <iostream>, <vector>, <queue>, and <omp.h>.
//Graph Structure: A structure Graph is defined to represent a graph. It contains the number of vertices (V) and an adjacency list (adj) to store the connections between vertices.
//addEdge Function: This function addEdge adds an edge between two vertices in the graph by pushing the destination vertex into the adjacency list of the source vertex.
//parallelBFS Function: This is the parallel Breadth First Search function. It takes the graph, the start vertex, and a vector to store distances as input parameters.
//It initializes a vector visited to keep track of visited vertices and marks all vertices as not visited initially.
//It creates a queue queue to perform BFS.
//It marks the start vertex as visited, sets its distance to 0, and enqueues it.
//Inside the main loop, it dequeues a vertex from the queue and explores its neighbors.
//The loop that visits neighbors is parallelized using OpenMP's #pragma omp parallel for directive. This allows multiple threads to visit different neighbors concurrently.
//If a neighbor is not visited, it marks it as visited, updates its distance, and enqueues it.
//Main Function:
//It initializes a graph with 6 vertices and adds example edges between them.
//It creates a vector distances to store distances from the start vertex to each vertex in the graph, initialized with -1 (indicating unreachable).
//It specifies the start vertex for BFS (vertex 0).
//It calls the parallelBFS function to perform BFS starting from the specified start vertex.
//After BFS is completed, it prints the distances from the start vertex to each vertex in the graph.
//Overall, this code performs parallel BFS on the given graph, starting from a specified vertex, and computes the distances from the start vertex to every other vertex in the graph. It demonstrates the use of OpenMP to parallelize the traversal process, potentially improving performance on multi-core processors.
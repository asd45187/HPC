#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>

using namespace std;

struct Graph {
    int V;
    vector<vector<int>> adj;
};

void addEdge(Graph& graph, int src, int dest) {
    graph.adj[src].push_back(dest);
}

void parallelDFS(Graph& graph, int vertex, vector<bool>& visited) {
    visited[vertex] = true;

    #pragma omp parallel for
    for (int i = 0; i < graph.adj[vertex].size(); ++i) {
        int neighbor = graph.adj[vertex][i];
        if (!visited[neighbor]) {
            parallelDFS(graph, neighbor, visited);
        }
    }
}

int main() {
    Graph graph;
    int numVertices = 6;
    graph.V = numVertices;
    graph.adj.resize(numVertices);

    addEdge(graph, 0, 1);
    addEdge(graph, 0, 2);
    addEdge(graph, 1, 3);
    addEdge(graph, 1, 4);
    addEdge(graph, 2, 5);

    vector<bool> visited(numVertices, false);

    int startVertex = 0;
    parallelDFS(graph, startVertex, visited);

    return 0;
}


// 0
 //1
//2
//3
//4 
//5 

// Explain



// 1. **Header Includes**:
//    - `#include <iostream>`: Provides input/output operations.
//    - `#include <vector>`: Provides the vector container class.
//    - `#include <stack>`: Provides the stack container class.
//    - `#include <omp.h>`: Provides OpenMP functionalities for parallelism.

// 2. **Graph Structure**:
//    - `struct Graph`: Represents a graph using an adjacency list.
//    - It has an integer `V` representing the number of vertices and a 2D vector `adj` to store adjacency lists for each vertex.

// 3. **Utility Functions**:
//    - `void addEdge(Graph& graph, int src, int dest)`: Adds an edge between two vertices in the graph.
//    - It appends the destination vertex `dest` to the adjacency list of the source vertex `src`.

// 4. **Parallel Depth First Search Function**:
//    - `void parallelDFS(Graph& graph, int vertex, vector<bool>& visited)`: Performs parallel Depth First Search (DFS) starting from a given vertex.
//    - It marks the current vertex as visited.
//    - It parallelizes the loop that iterates over the neighbors of the current vertex using OpenMP's `#pragma omp parallel for` directive.
//    - If a neighbor vertex has not been visited, it recursively calls `parallelDFS` on that neighbor vertex.

// 5. **Main Function**:
//    - Initializes a graph with 6 vertices and adds edges between them.
//    - Creates a vector `visited` to keep track of visited vertices, initialized with `false` for all vertices.
//    - Specifies the start vertex (`startVertex`) for the DFS traversal.
//    - Calls the `parallelDFS` function to perform parallel DFS starting from the specified start vertex.

// 6. **Explanation**:
//    - The code demonstrates how to implement parallel Depth First Search (DFS) using OpenMP in a graph represented by an adjacency list.
//    - Parallelism is introduced by parallelizing the loop that iterates over the neighbors of each vertex in the `parallelDFS` function.
//    - OpenMP's parallel for directive (`#pragma omp parallel for`) allows multiple threads to explore different branches of the DFS tree concurrently.
//    - By leveraging parallelism, the code aims to improve the performance of DFS traversal, especially for large graphs, by utilizing multiple threads to explore different parts of the graph simultaneously.
//    - The main function initializes the graph, specifies the start vertex, and invokes the parallel DFS function to explore the graph in parallel.
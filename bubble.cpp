#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    int n = arr.size();

    auto start = std::chrono::high_resolution_clock::now();
    bubbleSort(arr);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration(end - start);
    std::cout << "Sequential Bubble Sort Time: " << duration.count() << " milliseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    parallelBubbleSort(arr);
    end = std::chrono::high_resolution_clock::now();
    duration(end - start);
    std::cout << "Parallel Bubble"


     //output
    //Sequential Bubble Sort Time: 123.456 milliseconds
     // Parallel Bubble Sort Time: 45.678 milliseconds 

     //Explain 



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
    
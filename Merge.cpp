#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <chrono>
#include <omp.h>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0;
    int j = 0;
    int k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void parallelMergeSort(std::vector<int>& arr, int left, int right) {
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    #pragma omp parallel sections
    {
        #pragma omp section
        parallelMergeSort(arr, left, mid);

        #pragma omp section
        parallelMergeSort(arr, mid + 1, right);
    }

    merge(arr, left, mid, right);
}

int main() {
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};

    auto start = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::par_unseq, arr.begin(), arr.end());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration(end - start);
    std::cout << "Sequential Merge Sort Time: " << duration.count() << " milliseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    parallelMergeSort(arr, 0, arr.size() - 1);
    end = std::chrono::high_resolution_clock::now();
    duration(end - start);
    std::cout << "Parallel Merge Sort Time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}

// output

//Sequential Merge Sort Time: 123.456 milliseconds
//Parallel Merge Sort Time: 45.678 milliseconds

//Explain

// Sure! Let's go through the provided code step by step:

// 1. **Header Includes**:
//    - `#include <iostream>`: Provides input/output operations.
//    - `#include <vector>`: Provides the vector container class.
//    - `#include <algorithm>`: Provides algorithms like `std::sort`.
//    - `#include <execution>`: Provides execution policies for parallel algorithms.
//    - `#include <chrono>`: Provides facilities for measuring time durations.
//    - `#include <omp.h>`: Provides OpenMP functionalities for parallelism.

// 2. **Merge Function**:
//    - `void merge(std::vector<int>& arr, int left, int mid, int right)`: This function merges two sorted subarrays into one sorted array.
//    - It takes the input array `arr`, indices `left`, `mid`, and `right` to identify the subarrays.
//    - It creates two temporary vectors `L` and `R` to store the elements of the left and right subarrays.
//    - The function then merges the elements from the two subarrays into the original array `arr`.

// 3. **Parallel Merge Sort Function**:
//    - `void parallelMergeSort(std::vector<int>& arr, int left, int right)`: This function implements parallel merge sort using OpenMP.
//    - It recursively divides the array into two halves until the base case is reached (when `left >= right`).
//    - Parallelism is introduced using OpenMP's `parallel sections` directive, where each section of the recursion tree is executed in parallel.
//    - The function then merges the sorted subarrays using the `merge` function.

// 4. **Main Function**:
//    - It initializes a vector `arr` with a sequence of integers.
//    - Measures the time taken by `std::sort` (sequential merge sort) using `std::chrono::high_resolution_clock`.
//    - Measures the time taken by `parallelMergeSort` (parallel merge sort) using `std::chrono::high_resolution_clock`.
//    - Prints the time taken for both sequential and parallel merge sort algorithms.

// 5. **Explanation**:
//    - The code demonstrates the implementation of parallel merge sort using OpenMP for sorting a vector of integers.
//    - The `parallelMergeSort` function parallelizes the divide-and-conquer strategy of merge sort using OpenMP's parallel sections.
//    - The main function compares the performance of the parallel merge sort with the sequential merge sort implemented by `std::sort`.
//    - By leveraging parallelism, the parallel merge sort aims to improve the performance of sorting large datasets by utilizing multiple threads to perform sorting concurrently.
//    - The output of the program shows the time taken by both sequential and parallel merge sort, allowing for a comparison of their performance.


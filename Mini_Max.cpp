#include <iostream>
#include <vector>
#include <numeric>
#include <execution>
#include <omp.h>

int min(const std::vector<int>& arr) {
    int minVal = *std::min_element(std::execution::par_unseq, arr.begin(), arr.end());
    return minVal;
}

int max(const std::vector<int>& arr) {
    int maxVal = *std::max_element(std::execution::par_unseq, arr.begin(), arr.end());
    return maxVal;
}

int sum(const std::vector<int>& arr) {
    int sumVal = std::reduce(std::execution::par_unseq, arr.begin(), arr.end(), 0);
    return sumVal;
}

double average(const std::vector<int>& arr) {
    int sumVal = sum(arr);
    double avgVal = static_cast<double>(sumVal) / arr.size();
    return avgVal;
}

int main() {
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};

    std::cout << "Min: " << min(arr) << std::endl;
    std::cout << "Max: " << max(arr) << std::endl;
    std::cout << "Sum: " << sum(arr) << std::endl;
    std::cout << "Average: " << average(arr) << std::endl;

    return 0;
}

//output
//Min: 11
//Max: 90
//Sum: 238
//Average: 34

//explaination 
//Header Includes:
//#include <iostream>: Provides input/output operations.
// #include <vector>: Provides the vector container class.
// #include <numeric>: Provides numeric operations on sequences of values.
// #include <execution>: Provides execution policies for parallel algorithms.
// #include <omp.h>: Provides OpenMP functionalities for parallelism.
// Utility Functions:
// min(const std::vector<int>& arr): Finds the minimum element in the vector.
// max(const std::vector<int>& arr): Finds the maximum element in the vector.
// sum(const std::vector<int>& arr): Computes the sum of all elements in the vector.
// average(const std::vector<int>& arr): Computes the average of all elements in the vector.
// Parallel Execution Policies:
// The functions min, max, and sum use the par_unseq execution policy from the <execution> header. This policy allows the algorithms to execute in parallel and potentially out-of-order.
// Main Function:
// Creates a vector arr containing a sequence of integers.
// Calls the utility functions min, max, sum, and average to compute and print the minimum, maximum, sum, and average of the elements in the vector, respectively.
// Explanation:
// The min, max, and sum functions utilize parallel execution policies to potentially parallelize the computation of the minimum, maximum, and sum values, respectively, across multiple threads.
// The average function internally calls the sum function to compute the sum of elements and then divides it by the number of elements to calculate the average.
// The main function demonstrates the usage of these utility functions by applying them to a vector of integers and printing the results.
// Overall, this code showcases the use of parallel algorithms and execution policies provided by the C++ Standard Library to efficiently compute statistical properties of a sequence of values, potentially leveraging parallelism for improved performance.






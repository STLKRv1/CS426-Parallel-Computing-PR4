# CS426-Parallel-Computing-PR4
Simple power iteration method implemented for sparse matrices using CUDA 10.0

## To Compile  
cd into directory of cuda file  
```
nvcc powerIteration.cu
```

## To Run  
```
a.exe num_of_threads num_of_iterations print_param test/test.txt
```

**or**
(for windows)  
```
.\a.exe num_of_threads num_of_iterations print_param test/test.txt
```

**or**
(for linux)  
```
./a.out num_of_threads num_of_iterations print_param test/test.txt
```

***parameters:***   
**num_of_threads:** Number of cuda threads to be executed in a block
**num_of_iterations:** Number of iterations that is to be executed for power iterations ie. Number of matrix-vector multiplications  
**print_param:** 1: prints matrix in the sparse matrix form, initial vector(all 1's) and resulting vector after all the iteratons  
2:prints just the resulting vector after all the iterations.  
**test/test.txt:** directory path of test files in the format given in the project4.pdf(test folder contains tests however you can use your own matrices from matrixmarket etc.)

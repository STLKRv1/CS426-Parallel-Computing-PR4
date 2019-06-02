
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
//#include <time.h>
//#include <sys\timeb.h> 

#define EMPTY_LAST_ROW_PTR -1
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX_BLOCK_SIZE 512
#define NUM_BLOCKS 1

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

struct Sparse_matrix
{
    double* values;
    size_t num_cols;
    int* col_ptrs;
    size_t num_rows;
    int* row_ptrs;
};

struct Raw_matrix_tuple //used in read file stage
{
    int row;
    int col;
    double val;
};

__global__ void matrix_vector_mult_kernel(const double *x, const int *row_ptrs, const int *col_ptrs, const double *values, const int num_rows,
    const int num_cols, int total_num_threads, double *x_prime)
{
    int rows_per_thread = num_rows / total_num_threads;
    if (rows_per_thread < 1)
        rows_per_thread = 1;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * rows_per_thread;
    double sum = 0;

    if(idx < num_rows)
        for (int i = idx; i < ((threadIdx.x + blockIdx.x * blockDim.x == total_num_threads - 1) && (row_ptrs[i] != -1) ? num_rows : (idx + rows_per_thread)); i++)
        {
            sum = 0;
            //printf("matrix->row_ptrs[%d / %d]: %d\n", i, matrix.num_rows, matrix.row_ptrs[i]);
            for (int j = row_ptrs[i]; j != -1 && (j < ((i + 1 != num_rows) && row_ptrs[i + 1] != -1 ? row_ptrs[i + 1] : num_cols)); j++)
            {
                //printf("\ts_matrix->col_ptrs[%d/%d]: %d \n", j, matrix.num_cols, matrix.col_ptrs[j]);
                //printf("\t[%d][%d]: %lf \n", i, matrix.col_ptrs[j], matrix.values[j]);
                sum += values[j] * x[col_ptrs[j]];
            }
            x_prime[i] = sum;
        }
}

__host__ int comparator(const void *p, const void *q)
{
    int l = ((Raw_matrix_tuple *)p)->row;
    int r = ((Raw_matrix_tuple *)q)->row;

    if (l == r)
    {
        return ((Raw_matrix_tuple *)p)->col - ((Raw_matrix_tuple *)q)->col;
    }
    else
        return l - r;
}

int num_row = 0, num_col = 0, num_non_zero = 0;
Sparse_matrix matrix;

__host__ void read_matrix(const char* file_name, Sparse_matrix* s_matrix)
{
    FILE* file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Unable to open file!\n");
        return;
    }

    //read #row, #col #val
    fscanf(file, "%d %d %d", &num_row, &num_col, &num_non_zero);

    //read row col value triplets
    int i = 0;
    Raw_matrix_tuple* matrix_elements = (Raw_matrix_tuple*)calloc(num_non_zero, sizeof(Raw_matrix_tuple));
    while (EOF != fscanf(file, "%d %d %le\n", &matrix_elements[i].row, &matrix_elements[i].col, &matrix_elements[i].val))
    {
        //printf("%d %d %f\n", matrix_elements[i].row, matrix_elements[i].col, matrix_elements[i].val);
        matrix_elements[i].row--;
        matrix_elements[i].col--;
        i++;
    }
    fclose(file);

    //sort row increasing order
    qsort((void*)matrix_elements, num_non_zero, sizeof(Raw_matrix_tuple), comparator);

    /*s_matrix->num_rows = 1;
    s_matrix->num_cols = 1;
    //calculate and allocate space sparse matrix
    for (int i = 1; i < num_non_zero; i++)
    {
        //printf("%d %d %f\n", data[i].row, data[i].col, data[i].val);
        if (matrix_elements[i].row != matrix_elements[i - 1].row)
        {
            s_matrix->num_rows++;
            s_matrix->num_cols++;
        }
        else if(matrix_elements[i].col != matrix_elements[i - 1].col)
        {
            s_matrix->num_cols++;
        }
    }*/
    s_matrix->num_rows = num_row;
    s_matrix->num_cols = num_non_zero;
    s_matrix->values = (double*)calloc(s_matrix->num_cols, sizeof(double));
    s_matrix->col_ptrs = (int*)calloc(s_matrix->num_cols, sizeof(int));
    s_matrix->row_ptrs = (int*)calloc(s_matrix->num_rows, sizeof(int));

    s_matrix->row_ptrs[0] = 0;
    //printf("if: %d %d\n", matrix_elements[num_non_zero - 1].row, num_row-1);
    if (matrix_elements[num_non_zero - 1].row != num_row - 1)
        s_matrix->row_ptrs[s_matrix->num_rows - 1] = EMPTY_LAST_ROW_PTR;
    int data_index = 0, col_index = 0;

    //decode into sparse matrix
    for (int row_index = 0; row_index < num_row; row_index++)
    {
        //if(col_index != s_matrix->num_cols)
        s_matrix->row_ptrs[row_index] = col_index;

        //printf("s_matrix->row_ptrs[%d / %d]: %d\n", row_index, s_matrix->num_rows, s_matrix->row_ptrs[row_index]);
        //printf("if %d, %d\n", data[data_index].row, row_index);
        if (matrix_elements[data_index].row == row_index)
        {
            do
            {
                //printf("\ttriplet: %d %d\n", matrix_elements[data_index].row, matrix_elements[data_index].col);
                s_matrix->col_ptrs[col_index] = matrix_elements[data_index].col;
                s_matrix->values[col_index] = matrix_elements[data_index].val;
                data_index++;
                //printf("\ts_matrix->col_ptrs[%d/%d]: %d \n",col_index, s_matrix->num_cols, s_matrix->col_ptrs[col_index]);
                //printf("\t[%d][%d]: %lf \n", row_index, matrix.col_ptrs[col_index], matrix.values[col_index]);
                col_index++;
            } while (matrix_elements[data_index - 1].row == matrix_elements[data_index].row);
        }

    }
    free(matrix_elements);
}

__host__ void print_vector(double* vector, size_t size)
{
    printf("[ ");
    for (size_t i = 0; i < size; i++)
    {
        printf("%lf", vector[i]);
        if (i != size - 1)
            printf(", ");
        else
            printf(" ]\n");
    }
}

__host__ void print_matrix(Sparse_matrix matrix)
{
    printf("values_array: \n");
    /*printf("[ ");
    for (size_t i = 0; i < matrix.num_cols; i++)
    {
        printf("%lf", matrix.values[i]);
        if (i != matrix.num_cols - 1)
            printf(", ");
        else
            printf(" ]");
    }*/
    print_vector(matrix.values, matrix.num_cols);

    printf("col_indices:\n");
    printf("[ ");
    for (size_t i = 0; i < matrix.num_cols; i++)
    {
        printf("%d", matrix.col_ptrs[i]);
        if (i != matrix.num_cols - 1)
            printf(", ");
        else
            printf(" ]\n");
    }

    printf("row_ptr: \n");
    printf("[ ");
    for (size_t i = 0; i < matrix.num_rows; i++)
    {
        printf("%d", matrix.row_ptrs[i]);
        if (i != matrix.num_rows - 1)
            printf(", ");
        else
            printf(" ]\n");
    }
}

double* x_vector;
int num_iter = 1;
int num_threads;
int print;

double* d_values;
int* d_col_ptrs;
int* d_row_ptrs;
double* d_x_vector;
double* d_x_prime;

int main(int argc, char *argv[])
{
    char *file_name;

    if (argc > 3)
    {
        num_threads = atoi(argv[1]);
        num_iter = atoi(argv[2]);
        print = atoi(argv[3]);
        file_name = argv[4];
    }
    else
        return 0;
    //File read
    read_matrix(file_name, &matrix);

    if (print == 1)
        printf("vector:\n");

    //Init x_vector all 1s
    x_vector = (double*)calloc(num_col, sizeof(double));
    for (size_t i = 0; i < num_col; i++)
    {
        x_vector[i] = 1.0;
        /*if(print)
            printf("[%d]: %lf\n", i, x_vector[i]);*/
    }

    if (print == 1)
        print_vector(x_vector, num_col);

    /*FILE * fp;
    fp = fopen("test/2.txt", "w");*/

    if (print == 1)
    {
        printf("\ninitial matrix: \n");
        /*for (int i = 0; i < matrix.num_rows; i++)
        {
            //double sum = 0;
            //printf("matrix->row_ptrs[%d / %d]: %d\n", i, matrix.num_rows, matrix.row_ptrs[i]);
            for (int j = matrix.row_ptrs[i]; j!=-1 && (j < ( (i + 1 != matrix.num_rows) && matrix.row_ptrs[i + 1] !=-1 ? matrix.row_ptrs[i + 1] : matrix.num_cols)); j++)
            {
                //printf("\ts_matrix->col_ptrs[%d/%d]: %d \n", j, matrix.num_cols, matrix.col_ptrs[j]);
                printf("[%d][%d]: %lf \n", i, matrix.col_ptrs[j], matrix.values[j]);
                //sum += matrix.values[j] * x_vector[matrix.col_ptrs[j]];
            }
            //fprintf(fp, "[%d] :%lf\n",i, sum);
        }*/
        print_matrix(matrix);
    }
    //fclose(fp);

    //Allocate device Arrays
    cudaMalloc(&d_row_ptrs, matrix.num_rows * sizeof(int));
    cudaMalloc(&d_col_ptrs, matrix.num_cols * sizeof(int));
    cudaMalloc(&d_values, matrix.num_cols * sizeof(double));
    cudaMalloc(&d_x_vector, num_col * sizeof(double));
    cudaMalloc(&d_x_prime, num_col * sizeof(double));

    //printf("1: %s\n", cudaGetErrorString (cudaGetLastError()));

    //Init device arrays
    cudaMemcpy(d_row_ptrs, matrix.row_ptrs, matrix.num_rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptrs, matrix.col_ptrs, matrix.num_cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, matrix.values, matrix.num_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_vector, x_vector, num_col * sizeof(double), cudaMemcpyHostToDevice);

    //printf("2: %s\n", cudaGetErrorString (cudaGetLastError()));

    //Calculate the block and thread dims
    /*int threads_per_block, num_blocks;
    if (num_threads > MAX_BLOCK_SIZE)
    {
        threads_per_block = num_threads / MAX_BLOCK_SIZE;
        num_blocks = MAX_BLOCK_SIZE / num_threads;
    }
    else
    {
        threads_per_block = num_threads;
        num_blocks = 1;
    }*/
    //printf("%d %d\n", num_blocks, threads_per_block);

    //Kernel call
    //clock_t start = clock();
    /*struct timeb start, end;
    ftime(&start);
    int diff = 0;*/
    for (size_t iter = 0; iter < num_iter; iter++)
    {
        if (iter % 2)
        {
            matrix_vector_mult_kernel << < NUM_BLOCKS, num_threads >> > (d_x_prime, d_row_ptrs, d_col_ptrs, d_values, matrix.num_rows, matrix.num_cols, num_threads, d_x_vector);
            //printf("vector\n");
        }
        else
        {
            matrix_vector_mult_kernel << < NUM_BLOCKS, num_threads >> > (d_x_vector, d_row_ptrs, d_col_ptrs, d_values, matrix.num_rows, matrix.num_cols, num_threads, d_x_prime);
            //printf("prime\n");
        }
        cudaDeviceSynchronize();
    }
    //clock_t end = clock();
    /*ftime(&end);
    diff = (int)(1000.0 * (end.time - start.time)
        + (end.millitm - start.millitm));*/

    //printf("parallel time: %lf seconds\n", (double)(end - start)/CLOCKS_PER_SEC);
    //printf("parallel time: %u milliseconds\n", diff);
    //printf("3: %s\n", cudaGetErrorString (cudaGetLastError()));

    //Copy back the results
    if (num_iter % 2)
    {
        cudaMemcpy(x_vector, d_x_prime, num_col * sizeof(double), cudaMemcpyDeviceToHost);
        //printf("prime\n");
    }
    else
    {
        cudaMemcpy(x_vector, d_x_vector, num_col * sizeof(double), cudaMemcpyDeviceToHost);
        //printf("vector\n");
    }
    //printf("4: %s\n", cudaGetErrorString (cudaGetLastError()));
    if (print != 0)
    {
        printf("\nresulting vector: \n");
        print_vector(x_vector, num_col);


    }

    cudaFree(d_col_ptrs);
    cudaFree(d_row_ptrs);
    cudaFree(d_values);
    cudaFree(d_x_prime);
    cudaFree(d_x_vector);

    free(x_vector);
    free(matrix.values);
    free(matrix.col_ptrs);
    free(matrix.row_ptrs);
}

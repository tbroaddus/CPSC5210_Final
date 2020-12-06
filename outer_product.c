/**
 * Outer product algorithm for 1-D SpGEMM 
 * 
 * Using MPI_Reduce to compute local portions of matrix product for each rank(process)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>


// Used for debugging purposes  
void printMatrix(char* name, float** matrix, const int rows, const int cols);


// main
/**
 * @param N_rows The N number of rows in the matrix product
 * total number of elements = N * N (square matrix)
 */
int main(int argc, char* argv[]) {

  // Setup 
  int rank, world_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc != 2) {
    if (rank == 0) {
      fprintf(stderr, "Error: Provide row size for square matrix\n");
    }
    exit(1);
  }

  // Const values for dimensions
  // All row and col values are the same since we are working with square matrices
  // Element values are the same as well
  const int a_rows = atoi(argv[1]);
  const int a_cols = a_rows;
  const int n_elements = a_rows * a_cols;
  const int b_rows = a_cols;
  const int b_cols = a_rows;
  const int c_rows = a_rows;
  const int c_cols = b_cols;
  const int c_elements = c_rows * c_cols;

  // Matrix A: Dynamically allocating 2-D matrix contiguously, setting all values to 1.0
  float* mem = malloc( n_elements * sizeof(float));
  for (size_t i = 0; i < n_elements; i++) {
    mem[i] = 1.0;
  }
  float** A = malloc(a_rows * sizeof(float*));
  A[0] = mem;
  for (int i = 1; i < a_rows; i++) {
    A[i] = A[i-1] + a_cols;
  }

  // Matrix B: Dynamically allocating 2-D matrix contiguously, setting all values to 1.0
  mem = malloc( n_elements * sizeof(float));
  for (size_t i = 0; i < n_elements; i++) {
    mem[i] = 1.0;
  }
  float** B = malloc(b_rows * sizeof(float*));
  B[0] = mem;
  for (int i = 1; i < b_rows; i++) {
    B[i] = B[i-1] + b_cols;
  }

  // Matrix C (matrix product): Dynamically allocating 2-D matrix contigously, all values set to 0.0 (calloc)
  mem = calloc(c_elements, sizeof(float));
  float** C = malloc(c_rows * sizeof(float*));
  C[0] = mem;
  for (int i = 1; i < c_rows; i++) {
    C[i] = C[i-1] + c_cols;
  }

  // Final Matrix to act as recv buffer; we cannot use the same pointer for both send and recv in MPI_Reduce
  // Dynamically allocating 2-D matrix contiguously, all values set to 0.0 (calloc)
  mem = calloc(c_elements, sizeof(float));
  float** Final = malloc(c_rows * sizeof(float*));
  Final[0] = mem;
  for (int i = 1; i < c_rows; i++) {
    Final[i] = Final[i-1] + c_cols;
  }

  mem = NULL;

  // Sync, start timer
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();

  // Calculations for local domain
  const int index_size = (int)ceil((double)a_cols / (double)world_size);
  const int start_index = index_size * rank;
  int index = start_index + index_size - 1;
  if (index > a_rows - 1) {
    index = a_rows - 1;
  }
  const int end_index = index;

  for (int i = start_index; i <= end_index; i++) {
    for (int j = 0; j < a_rows; j++) {
      for (int k = 0; k < b_cols; k++) { 
        C[j][k] += A[j][i] * B[i][k];
      }
    }
  }
  // Iterating through each rank to sum the partial products
  for (int i = 0; i < world_size; i++) {
    // Must determine size on the fly...
    // The last rank may contain less elements than the others
    int size;
    // if (ending_index > c_rows)
    if (((i * index_size) + index_size - 1) > c_rows - 1) {
      // size = c_cols * ((c_rows ) - starting_index) 
      size = c_cols * ((c_rows) - (i * index_size));
    } 
    // else; size = c_cols * (ending_index - starting_index)
    else {
      size = c_cols * ((i * index_size + index_size) - (i * index_size));
    }
    // MPI_Reduce(MPI_SUM) to calculate local portion for current rank i 
    MPI_Reduce(C[i * index_size], Final[i * index_size], size, MPI_FLOAT, MPI_SUM, i, MPI_COMM_WORLD);
  }
  // End of execution, get end time
  const double end = MPI_Wtime();

 
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1 * rank);
  printf("Rank: %d\n", rank);
  printMatrix("Partial C", Final, c_rows, c_cols);
  printMatrix("Partial C", C, c_rows, c_cols);


  
  

  // Total time for this rank 
  const double total = end - start;
  double exe_time;
  // Reduce to find MAX total time amongst ranks;
  MPI_Reduce(&total, &exe_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  // Print result
  if (rank == 0) {
    printf("\tMPI Reduce Outer Product Algorithm for Matrix Multiplication\n");
    printf("\tN rows\t\tExecution time (seconds)\n\n");
    printf("\t%d\t\t%f\n", c_rows, exe_time);
  }
  

  // Cleanup, finalize, and exit
  free(A[0]);
  free(A);
  free(B[0]);
  free(B);
  free(C[0]);
  free(C);
  free(Final[0]);
  free(Final);
  MPI_Finalize();   
  exit(0); 
}


void printMatrix(char* name, float** matrix, const int rows, const int cols) {
  printf("Matrix %s:\n", name);
  printf("\n");
  for (size_t i = 0; i < rows; i++) {
    printf("\t");
    for (size_t j = 0; j < cols; j++) {
      printf("%.2f  ", matrix[i][j]);
    }
    printf("\n\n");
  }
  printf("\n");
}

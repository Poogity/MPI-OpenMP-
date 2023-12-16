#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <malloc.h> // Include for _aligned_malloc

#define NUM_THREADS 4


int main(int argc, char** argv) {


	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	if (provided < MPI_THREAD_FUNNELED) {
		std::cerr << "Error: MPI threading level is not enough." << std::endl;
		MPI_Finalize();
		return 1;
	}
	int rank, size=3;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	omp_set_num_threads(NUM_THREADS);

	int n = 16384;
	if (argc == 2)
		n = atoi(argv[1]);


	const unsigned int start = n * rank / size;
	const unsigned int end = n * (rank + 1) / size;
	const unsigned int local_n = end - start;
	const unsigned int jstart = 0;
	const int alignment = 32;

	// allocate memory (aligned)
	double* A = (double*)_aligned_malloc(local_n * n * sizeof(double), alignment);
	double* v = (double*)_aligned_malloc(local_n * sizeof(double), alignment);
	double* w = (double*)_aligned_malloc(n * sizeof(double), alignment);

	double init_time_start = omp_get_wtime();

//#pragma omp simd collapse(2)
#pragma omp parallel for collapse(2)
	/// init A_ij = (i + 2*j) / n^2
	for (int i = 0; i < local_n; ++i)
		for (int j = 0; j < n; ++j)
			A[i * n + j] = ((i+start) + 2.0 * (j+jstart)) / (n * n);

//#pragma omp simd
#pragma omp parallel for
	/// init v_i = 1 + 2 / (i+0.5)
	for (int i = 0; i < local_n; ++i)
		v[i] = 1.0 + 2.0 / (i+start + 0.5);

//#pragma omp simd
#pragma omp parallel for
	/// init w_i = 1 - i / (3.*n)
	for (int j = 0; j < n; ++j)
		w[j] = 1.0 - (j+jstart) / (3.0 * n);
	double init_time_end = omp_get_wtime();

	if(rank==0) printf("Time to initialize values: %f\n", init_time_end - init_time_start);

	/// compute
	double start_time, end_time;
	double local_result = 0;
	double local_start_time = MPI_Wtime();

//#pragma omp simd collapse(2) reduction(+: local_result)
#pragma omp parallel for collapse(2) reduction(+: local_result)
	for (int i = 0; i < local_n; ++i) 
		for (int j = 0; j < n; ++j) 
			local_result += v[i] * A[i * n + j] * w[j];
		
	
	double result;
	MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	double local_end_time = MPI_Wtime();

	//get first thread start time
	MPI_Reduce(&local_start_time, &start_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	//get last thread finish time
	MPI_Reduce(&local_end_time, &end_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	if (rank == 0) {
		printf("Result = %lf\n", result);

		printf("Total MPI Processes: %d\n", size);
		printf("Total OpenMP Threads: %d\n", NUM_THREADS);

		/// free memory
		_aligned_free(A);
		_aligned_free(v);
		_aligned_free(w);

		printf("Work took %f seconds\n", end_time-start_time);
	}
	MPI_Finalize();
	return 0;
}

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

	const unsigned int local_m = n;

	const unsigned int start = n * rank / size;
	const unsigned int end = n * (rank + 1) / size;
	const unsigned int local_n = end - start;
	const unsigned int jstart = 0;
	const int alignment = 8;

	// allocate memory
	double* A = (double*)_aligned_malloc(local_n * local_m * sizeof(double), alignment);
	double* v = (double*)_aligned_malloc(local_n * sizeof(double), alignment);
	double* w = (double*)_aligned_malloc(local_m * sizeof(double), alignment);

	double simd_time_start = omp_get_wtime();

	/// init A_ij = (i + 2*j) / n^2
#pragma omp parallel for collapse(2)
	for (int i = 0; i < local_n; ++i)
		for (int j = 0; j < local_m; ++j)
			A[i * local_m + j] = ((i+start) + 2.0 * (j+jstart)) / (n * n);
#pragma omp parallel for
	/// init v_i = 1 + 2 / (i+0.5)
	for (int i = 0; i < local_n; ++i)
		v[i] = 1.0 + 2.0 / (i+start + 0.5);
#pragma omp parallel for
	/// init w_i = 1 - i / (3.*n)
	for (int j = 0; j < local_m; ++j)
		w[j] = 1.0 - (j+jstart) / (3.0 * n);
	double simd_time_end = omp_get_wtime();
	// printf("Time to initialize values: %f\n", simd_time_end - simd_time_start);

	/// compute
	double start_time, end_time;
	double local_start_time = MPI_Wtime();
	double local_result = 0;

	//double * global_v = (double*)malloc(n * sizeof(double));
	//MPI_Allgather(v, local_n, MPI_DOUBLE, global_v,local_n, MPI_DOUBLE, MPI_COMM_WORLD);

#pragma omp parallel for collapse(2) reduction(+: local_result)
	for (int i = 0; i < local_n; ++i) {
		for (int j = 0; j < local_m; ++j) {
			local_result += v[i] * A[i * local_m + j] * w[j];
		}
	}
	double result;
	MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	double local_end_time = MPI_Wtime();

	//get first thread start time
	MPI_Reduce(&local_start_time, &start_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	//get last thread finish time
	MPI_Reduce(&local_end_time, &end_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		//double result = quadratic_form_reduction(A, v, w, n, MPI_COMM_WORLD);
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

// Exercise 1, question 3: initial code

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <io.h>

#define sleep(x) Sleep(1000 * (x))

void do_work(int i) {
	printf("processing %d\n", i);
	//sleep(5);
}

int main(int argc, char** argv) {
	int rank;
	int size;
	MPI_Request req[2];
	MPI_Status status[2];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0)
		printf("Running with %d MPI processes\n", size);

	int M = 2;	// two tasks per process
	int input;


	for (int i = 0; i < M; i++) {
		MPI_Irecv(&input, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, &req[0]);
		if (rank == 0) {
			for (int j = 0; j < size; j++) {
				input = rand() % 1000;	// some random value
				MPI_Isend(&input, 1, MPI_INT, j % size, 100, MPI_COMM_WORLD, &req[1]);
			}
		}
		//ensure that process 0 sent the data before using it
		MPI_Wait(&req[0], &status[0]); 
		do_work(input);
	}
	
	MPI_Finalize();
	return 0;
}
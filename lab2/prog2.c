#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int nproc, rank;
    int number;
    int next, prev;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nproc < 2) {
        if (rank == 0) {
            printf("Need at least 2 processes for a ring.\n");
        }
        MPI_Finalize();
        return 0;
    }

    /* neighbors in the ring */
    next = (rank + 1) % nproc;
    prev = (rank - 1 + nproc) % nproc;

    if (rank == 0) {
        srand(time(NULL));
        number = rand() % 1000;  /* random number */

        printf("Process 0 created number %d, sending to process %d\n", number, next);
        MPI_Send(&number, 1, MPI_INT, next, 0, MPI_COMM_WORLD);

        /* receive it back from the last process */
        MPI_Recv(&number, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received number %d back from process %d, ring complete.\n",
               number, prev);

    } else {
        /* everyone else just receives from prev and sends to next */
        MPI_Recv(&number, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received number %d from process %d\n", rank, number, prev);

        MPI_Send(&number, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        printf("Process %d sent number %d to process %d\n", rank, number, next);
    }

    MPI_Finalize();
    return 0;
}

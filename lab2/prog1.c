#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int nproc, rank;
    int msg;  /* the message we send/receive */

    MPI_Init(&argc, &argv);               /* Initialize MPI */
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);/* number of processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* my rank */

    if (nproc < 2) {
        if (rank == 0) {
            printf("Need at least 2 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        msg = rank;  /* or any value you want to send */
        MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process %d sent value %d to process 1\n", rank, msg);
    } else if (rank == 1) {
        MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received value %d from process 0\n", rank, msg);
    } else {
        /* other ranks do nothing for this simple program */
        printf("Process %d is idle.\n", rank);
    }

    MPI_Finalize();
    return 0;
}

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#define MAXPROC 100 /* Max number of procsses */

int main(int argc, char* argv[]) {
    int i, nproc, rank, index;
    const int tag = 42;        /* Tag value for communication */
    const int root = 0;        /* Root process in broadcast */
    MPI_Status status;         /* Status object for non-blocking receive */
    MPI_Request recv_req[MAXPROC]; /* Request objects for non-blocking receive */
    char hostname[MAXPROC][MPI_MAX_PROCESSOR_NAME]; /* Received host names */
    char myname[MPI_MAX_PROCESSOR_NAME]; /* local host name string */
    int namelen;               // Length of the name

    /* Begin parallel region */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Rank --> rank
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); // Size --> nproc

    if (nproc > MAXPROC) {
        if (rank == 0) {
            fprintf(stderr, "Error: nproc(%d) > MAXPROC(%d)\n", nproc, MAXPROC);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Get hostname */
    MPI_Get_processor_name(myname, &namelen);  // MPI_Get_processor_name --> myname, namelen
    if (namelen >= MPI_MAX_PROCESSOR_NAME)
        namelen = MPI_MAX_PROCESSOR_NAME - 1;
    myname[namelen] = '\0';   /* Terminate local buffer */

    if (rank == 0) { /* Process 0 does this */
        int myid = rank;

        /* Broadcast a message containing the process id */
        MPI_Bcast(&myid, 1, MPI_INT, root, MPI_COMM_WORLD);

        /* Start non-blocking calls to receive messages from all other processes */
        for (i = 1; i < nproc; i++) {
            MPI_Irecv(hostname[i], MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                      MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_req[i]);
        }

        /* While the messages are delivered, we could do computations here */
        printf("I am a very busy professor.\n");

        /* Iterate to receive messages from all other processes and print their hostnames */
        for (i = 1; i < nproc; i++) {
            /*
             * Wait until at least one message has been received
             * Request array starts from element 1, because we don't receive
             * any message from process 0 (this process)
             */
            MPI_Waitany(nproc, recv_req, &index, &status);

            /* index is the position in recv_req that completed */
            printf("Received a message from process %d on %s\n",
                   status.MPI_SOURCE, hostname[index]);
        }
    }
    else { /* all other processes do this */
        int myid;

        /* Receive the broadcasted message from process 0 */
        MPI_Bcast(&myid, 1, MPI_INT, root, MPI_COMM_WORLD);

        /* Send local hostname to process 0 */
        MPI_Send(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                 root, tag, MPI_COMM_WORLD);
    }

    /* Finish by finalizing the MPI library */
    MPI_Finalize();
    exit(0);
}
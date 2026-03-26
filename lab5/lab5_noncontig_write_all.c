/* noncontiguous access with a single collective I/O function */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define FILESIZE 1048576
#define FILESIZE 1024
#define INTS_PER_BLK 1

int main(int argc, char **argv)
{
    int *buf, rank, nprocs, nints, bufsize;
    MPI_File fh;
    MPI_Datatype filetype;
    MPI_Status status;

    const char *filename = "lab5_output.dat";

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Each process gets an equally sized portion of the total file size */
    bufsize = FILESIZE / nprocs;
    buf = (int *)malloc(bufsize);
    if (!buf) {
        if (rank == 0) {
            fprintf(stderr, "Failed to allocate buffer\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Number of ints this rank will write */
    nints = bufsize / sizeof(int);

    /* Initialize buffer with a rank-specific pattern */
    memset(buf, 'A' + rank, nints * sizeof(int));

    /* Decide layout for noncontiguous file access:
     * Use MPI_Type_vector to describe num_blocks blocks of INTS_PER_BLK ints
     * separated by a stride in units of ints.
     *
     * For simplicity, choose num_blocks so that:
     *   nints = num_blocks * INTS_PER_BLK
     */
    int num_blocks = nints / INTS_PER_BLK;
    if (num_blocks < 1) num_blocks = 1;

    /* Stride (in units of MPI_INT) between starts of consecutive blocks
     * in the file. Here we interleave data from all processes:
     * rank 0, rank 1, ..., rank nprocs-1, then repeat.
     */
    int stride = nprocs * INTS_PER_BLK;

    MPI_Type_vector(num_blocks,         /* number of blocks per process    */
                    INTS_PER_BLK,       /* number of elements per block    */
                    stride,             /* stride between block starts     */
                    MPI_INT,            /* oldtype                         */
                    &filetype);         /* new derived filetype            */
    MPI_Type_commit(&filetype);

    /* Open the file collectively, create if it does not exist.
     * Use write-only as requested in the lab.
     */
    MPI_File_open(MPI_COMM_WORLD,
                  filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL,
                  &fh);

    /* Set the file view for each process.
     * - disp: we leave 0 bytes at the beginning
     * - etype: MPI_INT (basic unit of access)
     * - filetype: our custom strided layout
     *
     * The displacement between ranks is INTS_PER_BLK * sizeof(int) * rank,
     * so rank 0 writes the first block, rank 1 the second, etc.
     */
    MPI_Offset disp = rank * INTS_PER_BLK * sizeof(int);
    MPI_File_set_view(fh,
                      disp,             /* displacement in bytes          */
                      MPI_INT,          /* etype                          */
                      filetype,         /* filetype                       */
                      "native",         /* data representation            */
                      MPI_INFO_NULL);

    /* Collective MPI-IO write using the individual file pointers.
     * Each rank writes nints elements from its buffer.
     */
    MPI_File_write_all(fh,
                       buf,
                       nints,           /* total number of elements       */
                       MPI_INT,
                       &status);

    /* Close file collectively */
    MPI_File_close(&fh);

    /* Data type cleanup */
    MPI_Type_free(&filetype);

    /* Free buffer */
    free(buf);

    /* Finalize MPI */
    MPI_Finalize();

    return 0;
}

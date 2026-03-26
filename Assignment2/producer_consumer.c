#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Tags:
 *  - TAG_WORK:       producer -> broker (random number)
 *  - TAG_ACK:        broker  -> producer (work accepted into buffer)
 *  - TAG_REQ_WORK:   consumer -> broker (request work)
 *  - TAG_ABORT:      broker  -> producer/consumer (simulation over)
 *  - TAG_NOWORK:     broker  -> consumer (no work currently available)
 *  - TAG_CONSUMED:   consumer -> broker (final consumed_count)
 */

#define TAG_WORK        1
#define TAG_ACK         2
#define TAG_REQ_WORK    3
#define TAG_ABORT       4
#define TAG_NOWORK      5
#define TAG_CONSUMED    6

/* Simple CPU work: at least 1000 iterations of arithmetic */
static void do_compute(int base)
{
    volatile double x = (double)(base == 0 ? 1 : base);
    for (int i = 0; i < 1000; i++) {
        x = x * 1.000001 + 0.000001;
        if (x > 1e9) x *= 0.5;
    }
}

/* Decide which ranks are producers vs consumers.
 * Rank 0 is always the broker (neither producer nor consumer).
 * Among ranks 1..(size-1), first half are producers, rest are consumers. */
static int is_producer(int rank, int size)
{
    if (rank == 0) return 0;
    int non_broker = size - 1;
    int half = non_broker / 2;        /* floor */
    int idx = rank - 1;               /* 0..non_broker-1 */
    return idx < half;
}

int main(int argc, char **argv)
{
    int rank, size;
    double sim_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <simulation_time_seconds>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    sim_time = atof(argv[1]);
    if (sim_time <= 0.0) {
        if (rank == 0)
            fprintf(stderr, "Simulation time must be > 0.\n");
        MPI_Finalize();
        return 1;
    }

    int non_broker      = size - 1;
    int num_producers   = non_broker / 2;
    int num_consumers   = non_broker - num_producers;

    if (rank == 0) {
        /* ==================== BROKER (PROCESS 0) ==================== */

        /* Buffer size = 2 * num_producers (from writeup) */
        int buffer_capacity = 2 * num_producers;
        int *buffer         = (int *)malloc(buffer_capacity * sizeof(int));
        int buffer_head     = 0;  /* index of first valid element */
        int buffer_count    = 0;  /* number of valid elements */

        /* Queue of producers that sent work while buffer was full and
         * are waiting for an ACK. */
        int *pending_producers = (int *)malloc(non_broker * sizeof(int));
        int pending_count      = 0;

        double start_time = MPI_Wtime();
        int all_abort_done = 0;

        MPI_Status status;

        /* Main broker loop: run until time expires AND there is nothing
         * left to handle (no buffered work and no pending producers). */
        while (1) {
            int value;
            MPI_Recv(&value, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;
            int tag = status.MPI_TAG;

            double now      = MPI_Wtime();
            double elapsed  = now - start_time;
            int response_type = (elapsed >= sim_time) ? TAG_ABORT : -1;

            if (tag == TAG_WORK) {
                /* Work from a producer */
                if (buffer_count < buffer_capacity && response_type != TAG_ABORT) {
                    /* Space in buffer and still within time window:
                     * 1) insert work, 2) send ACK. */
                    int pos = (buffer_head + buffer_count) % buffer_capacity;
                    buffer[pos] = value;
                    buffer_count++;

                    MPI_Send(&value, 1, MPI_INT, src, TAG_ACK, MPI_COMM_WORLD);
                } else {
                    /* Either buffer is full or time is up. */
                    if (response_type == TAG_ABORT) {
                        MPI_Send(&value, 1, MPI_INT, src, TAG_ABORT, MPI_COMM_WORLD);
                    } else {
                        /* Buffer full, remember this producer to ACK later
                         * when space becomes available. */
                        if (pending_count < non_broker) {
                            pending_producers[pending_count++] = src;
                        }
                    }
                }
            } else if (tag == TAG_REQ_WORK) {
                /* Work request from a consumer */
                if (response_type == TAG_ABORT) {
                    /* Time up: send ABORT instead of work. */
                    MPI_Send(&value, 1, MPI_INT, src, TAG_ABORT, MPI_COMM_WORLD);
                } else if (buffer_count > 0) {
                    /* Send first element from buffer to consumer. */
                    int work_val = buffer[buffer_head];
                    buffer_head = (buffer_head + 1) % buffer_capacity;
                    buffer_count--;

                    MPI_Send(&work_val, 1, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);

                    /* If some producers are waiting for ACK because the
                     * buffer used to be full, we can ACK one now. */
                    if (pending_count > 0) {
                        int prod_rank = pending_producers[pending_count - 1];
                        pending_count--;

                        int dummy = 0;
                        MPI_Send(&dummy, 1, MPI_INT, prod_rank, TAG_ACK, MPI_COMM_WORLD);
                    }
                } else {
                    /* No work available: send NOWORK message. */
                    MPI_Send(&value, 1, MPI_INT, src, TAG_NOWORK, MPI_COMM_WORLD);
                }
            }

            /* After time is up, we allow one more "drain" phase:
             * consumers keep asking, broker serves remaining buffer and
             * ACKs pending producers, and sends ABORT when contacted
             * and there is no useful work. */
            if (elapsed >= sim_time && buffer_count == 0 && pending_count == 0) {
                all_abort_done = 1;
            }

            if (all_abort_done)
                break;
        }

        /* After broker finishes, gather consumed_count from consumers.
         * (Producers send 0.) */
        long total_consumed = 0;
        for (int r = 1; r < size; r++) {
            long local;
            MPI_Recv(&local, 1, MPI_LONG, r, TAG_CONSUMED,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_consumed += local;
        }

        printf("Total number of messages consumed: %ld\n", total_consumed);

        free(buffer);
        free(pending_producers);

    } else {
        /* =========== PRODUCERS AND CONSUMERS (RANK > 0) =========== */

        int producer = is_producer(rank, size);
        long consumed_count = 0;
        int have_prev = 0;
        int prev_val  = 0;

        /* Different random seed per rank */
        srand((unsigned)time(NULL) + rank * 1237);

        if (producer) {
            /* -------------------- PRODUCER CODE -------------------- */
            while (1) {
                int work = rand();

                /* 1. Send "Work" non-blocking to broker */
                MPI_Request req;
                MPI_Isend(&work, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, &req);

                /* 2. Random arithmetic computation */
                do_compute(work);

                /* 3. Wait for broker response (ACK or ABORT) */
                MPI_Wait(&req, MPI_STATUS_IGNORE);

                int resp;
                MPI_Status st;
                MPI_Recv(&resp, 1, MPI_INT, 0, MPI_ANY_TAG,
                         MPI_COMM_WORLD, &st);

                if (st.MPI_TAG == TAG_ABORT) {
                    /* ABORT from broker: exit producer loop */
                    break;
                }
                /* else TAG_ACK: loop continues */
            }

        } else {
            /* -------------------- CONSUMER CODE -------------------- */
            while (1) {
                int dummy = 0;

                /* 1. Send "Request Work" non-blocking */
                MPI_Request req;
                MPI_Isend(&dummy, 1, MPI_INT, 0, TAG_REQ_WORK,
                          MPI_COMM_WORLD, &req);

                /* 2. Do compute with previous random number if available */
                if (have_prev)
                    do_compute(prev_val);
                else
                    do_compute(rank);  /* arbitrary initial seed */

                MPI_Wait(&req, MPI_STATUS_IGNORE);

                /* 3. Wait for response from broker */
                int val;
                MPI_Status st;
                MPI_Recv(&val, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

                if (st.MPI_TAG == TAG_WORK) {
                    /* Received real work from broker */
                    prev_val  = val;
                    have_prev = 1;
                    consumed_count++;  /* 4. increment local counter */
                } else if (st.MPI_TAG == TAG_ABORT) {
                    /* ABORT means simulation time is over */
                    break;
                } else if (st.MPI_TAG == TAG_NOWORK) {
                    /* No work for now, continue loop */
                    continue;
                }
            }
        }

        /* Send local consumed_count to broker at the very end.
         * Producers will just send 0 (never incremented). */
        MPI_Send(&consumed_count, 1, MPI_LONG, 0, TAG_CONSUMED, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

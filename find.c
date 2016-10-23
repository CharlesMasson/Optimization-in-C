#define _DEFAULT_SOURCE
#define NB_THREADS 8
#define CACHE_LINE_SIZE 64
#include <immintrin.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Written by Charles MASSON

/*
count_ones_table
----------------
This is an array of integers such that count_ones_table[i] is the number of ones in the binary value of i.
count_ones_table is used in the vector computing implementation to improve the performances.
*/

int count_ones_table[256];

int count_ones(int n) {
    int count = 0;
    while (n != 0) {
        if (n & 1)
            count++;
        n >>= 1;
    }
    return count;
}

void initialize_count_ones_table() {
    for (int i = 0; i < 256; i++)
        count_ones_table[i] = count_ones(i);
}

/*
get_time_ns()
-------------
Used to measure the execution time.
*/

long get_time_ns() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_nsec + t.tv_sec * 1E9L;
}

/*
generate_U()
------------
Generates an array of random integers.
*/

int* generate_U(int nb_items, int min_val, int max_val) {
    int *U = (int*) aligned_alloc(CACHE_LINE_SIZE, nb_items * sizeof(int));
    for (int i = 0; i < nb_items; i++)
        U[i] = rand() % (max_val - min_val + 1) + min_val;
    return U;
}

/*
Scalar implementation
---------------------
This is the basic implementation, which looks for all occurrences of val within U between indices i_start and i_stop,
with step i_step, and writes the indices of the occurrences of val in ind_val.
*/

int find(int *U, int i_start, int i_end, int i_step, int val, int **ind_val) {

    *ind_val = (int*) malloc(0);
    int nb_find = 0;

    for (int i = i_start; i <= i_end; i += i_step)
        if (U[i] == val) {
            nb_find++;
            *ind_val = (int*) realloc(*ind_val, nb_find * sizeof(int));
            (*ind_val)[nb_find - 1] = i;
        }

    return nb_find;
}

/*
Vector computing implementation
-------------------------------
This is the version that makes the most of vector computing.
i_step must be a multiple of 8.
*/

int vect_find(int *U, int i_start, int i_end, int i_step, int val, int **ind_val) {

    if (i_step % 8 != 0)
        return -1;

    *ind_val = (int*) malloc(0);
    int nb_find = 0;
    __m256 vect_val = _mm256_castsi256_ps(_mm256_set1_epi32(val));

    int i, j, mask;
    for (i = i_start; i + 8 < i_end; i += i_step) {
        mask = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps((float*) (U + i)), vect_val, _CMP_EQ_OS));
        if (mask) {
            *ind_val = (int*) realloc(*ind_val, (nb_find + count_ones_table[mask]) * sizeof(int));
            for (j = i; mask != 0; j++) {
                if (mask & 1) {
                    (*ind_val)[nb_find] = j;
                    nb_find++;
                }
                mask >>= 1;
            }
        }
    }

    for (; i <= i_end; i++)
        if (U[i] == val) {
            nb_find++;
            *ind_val = (int*) realloc(*ind_val, nb_find * sizeof(int));
            (*ind_val)[nb_find - 1] = i;
        }

    return nb_find;
}

/*
Multithreaded implementation
----------------------------
*/

// To avoid global variables, we could incorporate them to thread_data and feed them to watch_nb_find_thread.
bool stop_threads = false;
int nb_find_thread[NB_THREADS];
int *U_threads;
int val_threads;

// Those variables are specific to each thread.
struct thread_data {
    int i_start;
    int i_end;
    int i_step;
    int **ind_val;
    int *nb_find;
};

void *scalar_thread_function(void* thread_arg) {

    struct thread_data *my_data;
    my_data = (struct thread_data*) thread_arg;
    int i_start = my_data->i_start;
    int i_end = my_data->i_end;
    int i_step = my_data->i_step;
    int **ind_val = my_data->ind_val;
    int *nb_find = my_data->nb_find;

    *ind_val = (int*) malloc(0);

    for (int i = i_start; i <= i_end; i += i_step) {
        if (stop_threads)
            pthread_exit(NULL);
        if (U_threads[i] == val_threads) {
            (*nb_find)++;
            *ind_val = (int*) realloc(*ind_val, *nb_find * sizeof(int));
            (*ind_val)[*nb_find - 1] = i;
        }
    }

    pthread_exit(NULL);
}

void *vect_thread_function(void* thread_arg) {

    struct thread_data *my_data;
    my_data = (struct thread_data*) thread_arg;
    int i_start = my_data->i_start;
    int i_end = my_data->i_end;
    int i_step = my_data->i_step;
    int **ind_val = my_data->ind_val;
    int *nb_find = my_data->nb_find;

    *ind_val = (int*) malloc(0);
    __m256 vect_val = _mm256_castsi256_ps(_mm256_set1_epi32(val_threads));

    int i, j, mask;
    for (i = i_start; i + 8 < i_end; i += i_step) {
        if (stop_threads)
            pthread_exit(NULL);
        mask = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps((float*) (U_threads + i)), vect_val, _CMP_EQ_OS));
        if (mask) {
            *ind_val = (int*) realloc(*ind_val, (*nb_find + count_ones_table[mask]) * sizeof(int));
            for (j = i; mask != 0; j++) {
                if (mask & 1) {
                    (*ind_val)[*nb_find] = j;
                    (*nb_find)++;
                }
                mask >>= 1;
            }
        }
    }

    for (; i <= i_end; i++)
        if (U_threads[i] == val_threads) {
            (*nb_find)++;
            *ind_val = (int*) realloc(*ind_val, *nb_find * sizeof(int));
            (*ind_val)[*nb_find - 1] = i;
        }

    pthread_exit(NULL);
}


void *watch_nb_find(void* thread_arg) {

    int *k = (int*) thread_arg;
    int nb_find;

    while (true) {
        usleep(1000);
        if (stop_threads)
            pthread_exit(NULL);
        nb_find = 0;
        for (int t = 0; t < NB_THREADS; t++)
            nb_find += nb_find_thread[t];
        if (nb_find > *k) {
            stop_threads = true;
            pthread_exit(NULL);
        }
    }
}

int thread_find(int *U, int i_start, int i_end, int i_step, int val, int **ind_val, int k, int ver) {

    // Check the version to use (scalar of vector computing)
    void *thread_function;
    switch (ver) {
        case 0:
            thread_function = scalar_thread_function;
            break;
        case 1:
            if (i_step % 8 != 0)
                return -1;
            thread_function = vect_thread_function;
            break;
        default:
            printf("Invalid value for \"ver\".\n");
            return -1;
    }

    // Initialize the variables
    U_threads = U;
    val_threads = val;
    int **ind_val_thread[NB_THREADS];
    for (int t = 0; t < NB_THREADS; t++) {
        nb_find_thread[t] = 0;
        ind_val_thread[t] = (int**) malloc(sizeof(int*));
    }

    // Start watch_nb_find_thread if nescessary
    pthread_t watch_nb_find_thread;
    if (k >= 0) {
        pthread_create(&watch_nb_find_thread, NULL, watch_nb_find, &k);
    }

    // Initialize the variables that are specific to the threads and start the threads
    pthread_t threads[NB_THREADS];
    struct thread_data thread_data_array[NB_THREADS];
    for (int t = 0; t < NB_THREADS; t++) {
        thread_data_array[t].i_start = t * ((i_end - i_start) / i_step + 1) / NB_THREADS * i_step + i_start;
        thread_data_array[t].i_end = (t + 1) * ((i_end - i_start) / i_step + 1) / NB_THREADS * i_step + i_start - 1;
        thread_data_array[t].i_step = i_step;
        thread_data_array[t].ind_val = ind_val_thread[t];
        thread_data_array[t].nb_find = &nb_find_thread[t];
        pthread_create(&threads[t], NULL, thread_function, (void*) &thread_data_array[t]);
    }

    // Wait for the threads to finish running
    int nb_find = 0;
    for (int t = 0; t < NB_THREADS; t++) {
        pthread_join(threads[t], NULL);
        nb_find += nb_find_thread[t];
    }

    // In case watch_nb_find_thread is still running
    stop_threads = true;

    // Concatenate the arrays that contain the indices of the found occurrences
    *ind_val = (int*) malloc(nb_find * sizeof(int));
    int *ind_val_current = *ind_val;
    for (int t = 0; t < NB_THREADS; t++) {
        memcpy(ind_val_current, *ind_val_thread[t], nb_find_thread[t] * sizeof(int));
        ind_val_current += nb_find_thread[t];
    }

    // If necessary, ignore the extra indices
    if (k >= 0) {
        if (nb_find > k) {
            *ind_val = (int*) realloc(*ind_val, k * sizeof(int));
            nb_find = k;
        }
        pthread_join(watch_nb_find_thread, NULL);
    }

    return nb_find;
}

int main(int argc, char *argv[]){

    srand((unsigned) time(NULL));
    initialize_count_ones_table();
    long t_start, t_end;
    int **ind_val = (int**) malloc(sizeof(int*));
    int nb_find;

    bool print_ind = argc >= 2 ? atoi(argv[1]) != 0 : false;
    int size = argc >= 3 ? atoi(argv[2]) : 1E9;
    int min = argc >= 5 ? atoi(argv[3]) : 0;
    int max = argc >= 5 ? atoi(argv[4]) : 100;
    int val = argc >= 6 ? atoi(argv[5]) : rand() % (max - min + 1) + min;

    printf("Creating a random input array with %i values between %i and %i...\n", size, min, max);
    int *U = generate_U(size, min, max);
    printf("Done.\nValue to seek: %i.\n", val);

    // Scalar implementation
    printf("\nRunning scalar version...\n");
    t_start = get_time_ns();
    nb_find = find(U, 0, size - 1, 1, val, ind_val);
    t_end = get_time_ns();
    printf("Done. Execution time: %liµs\n", (t_end - t_start) / 1000);
    printf("Found %i valid indices.\n", nb_find);
    if (print_ind) {
        printf("Valid indices: ");
        for (int i = 0; i < nb_find; i++)
            printf("%i ", (*ind_val)[i]);
        printf("\n");
    }

    // Vector computing implementation
    printf("\nRunning vector version...\n");
    t_start = get_time_ns();
    nb_find = vect_find(U, 0, size - 1, 8, val, ind_val);
    t_end = get_time_ns();
    printf("Done. Execution time: %liµs\n", (t_end - t_start) / 1000);
    printf("Found %i valid indices.\n", nb_find);
    if (print_ind) {
        printf("Valid indices: ");
        for (int i = 0; i < nb_find; i++)
            printf("%i ", (*ind_val)[i]);
        printf("\n");
    }

    // multithreaded implementation (with vector computing)
    printf("\nRunning multithreaded version...\n");
    t_start = get_time_ns();
    nb_find = thread_find(U, 0, size - 1, 8, val, ind_val, -1, 1);
    t_end = get_time_ns();
    printf("Done. Execution time: %liµs\n", (t_end - t_start) / 1000);
    printf("Found %i valid indices.\n", nb_find);
    if (print_ind) {
        printf("Valid indices: ");
        for (int i = 0; i < nb_find; i++)
            printf("%i ", (*ind_val)[i]);
        printf("\n");
    }
}

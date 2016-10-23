# Performance optimization in C using vector computing and multithreading

The purpose of this project is to optimize a basic program written in C through diverse optimizations, especially **vector computing**, **memory alignment** and **multithreading**.

The function to optimize looks for the occurrences of a specified integer value in a given array of integers, fills an array with the indices of those occurrences and returns the number of occurrences:

```
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
```

# Vector computing optimization

The processor that I used for this project (Intel Core i7-3610QM) supports the AVX (Advanced Vector Extensions) x86 instruction set architecture, which can work with 256-bit vectors. However, AVX cannot compare integer vectors (`__m256i`). Then, there are two solutions for the project:

- use 128-bit integer vectors (`__m128i`), on which we can apply equality tests, but which can only contain 4 32-bit integers,
- cast integers to floats and use 256-bit float vectors (`__m256`), which we can compare together.

I chose the second solution, which is the most efficient one. However, with this method, we cannot compare integers whose binary value matches the `NaN` float value, in which case, the equality test with `_mm256_cmp_ps` will not be correct no matter what comparison operand is used (`_CMP_EQ_OS`, `_CMP_EQ__OQ`, `_CMP_EQ__US` or `_CMP_EQ__UQ`). Those values are written as `X111111 1XXXXXXX XXXXXXXX XXXXXXXX`. In other words, the program will work for integers lower or equal to `0x7F800000`, i.e., `2139095040`.

The optimization basically involves generating `vect_val`, a 256-bit integer vector (`__m256`) that contains 8 times the `val` value. Then, we compare `vect_val` to a group of 8 consecutive integers in `U` with `_mm256_cmp_ps`. This returns a mask that contains the results of the equalities. If that mask equals `0`, there is nothing to do (none of the integers within the group equals `val`). Otherwise, we reallocate (`realloc`) `ind_val` only once based on the number of values that equal `val` within the group, which is done in constant time thanks to `count_ones_table`.

I also tried to pad `__m256` with a struct and make sure that `vect_val` is memory aligned, so that it is the only variable on the cache line but this did not bring any improvement in terms of performance. `vect_val` is in the stack and it is likely that the rest of its cache line contains data that is managed by the same thread, which is why accessing `vect_val` most likely does not require fetching operations from higher cache levels.

# Multithreading optimization

The multithreaded version parallelizes the scalar and the vector computing versions. The argument `ver` specifies which version to use: `0` for the scalar version, `1` for the vector computing version.

To improve the performances, each thread works with its own counter (see `nb_find_thread`). It is also possible to specify the number of occurrences to look for with the argument `k`. An auxiliary thread, namely `watch_nb_find_thread`, is responsible for regularly checking the thread counters and setting the boolean variable to `stop_threads` to true when the required number of occurrences has been found. There is no need to use any mutex because only one thread can increment each of the counters and even if a counter is incremented while `stop_threads` compute the total number of found occurrences, the number of found occurrences can only be higher than the one calculated. Finally, the main thread returns `k` found occurrences and ignores the extra ones.

To reach higher performances, it is better to make each thread work on a sequential part of `U` (i.e., one chunk for each thread), which lowers the risk of having several threads reading simultaneously the same cache line.

# Compiling and running

To compile (with `gcc`): `gcc -std=c11 -mavx -pthread -o find.out [-O3] find.c`

To run: `./find.out [print_ind] [size] [min max] [val]`, with:

- `print_ind`: `1` to display found indices, `0` otherwise (default: `0`),
- `size`: size of the generated array `U` (default: `1E9`),
- `min` and `max`: lower and upper bounds of the generated values in `U` (default: `0` and `100`),
- `val`: value to find in the array `U` (default: random).

# Performance tests

## Setup:

- **CPU**: Intel Core i7-3610QM
    - 4 cores with Hyper-Threading (8 logical cores)
    - 2,3 GHz (turbo : 3,3 GHz)
    - instruction set: AVX
    - cache line size: 64 bytes
- **RAM**: 8GB, DDR3 (1600Mhz)

## Input:

- Array of size 1E9 with values between 0 and 100
- Value to find random between 0 and 100

## Compiler

`gcc` with or without all optimizations enabled (`-O3`). Intel's compiler `icc` would probably give better results, though.

## Results

| Implementation                        | Execution time without `-O3` | Execution time with `-O3` |
|---------------------------------------|------------------------------|---------------------------|
| Scalar                                | 2558 ms                      | 845 ms                    |
| Vector computing                      | 1074 ms                      | 587 ms                    |
| Multithreaded (with vector computing) | 337 ms                       | 249 ms                    |

The most efficient optimization is more than 10 times faster than the basic implementation (scalar without compilation optimizations).

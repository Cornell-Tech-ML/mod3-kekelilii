# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


## Task 3.1 NUMBA diagnostics
# Map
Parallel loop listing for  Function tensor_map.<locals>._map, /Users/kd/Library/Mobile Documents/com~apple~CloudDocs/Cornell Tech/2024Fall/MLE/mod3-kekelilii/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                  | 
        out: Storage,                                                                          | 
        out_shape: Shape,                                                                      | 
        out_strides: Strides,                                                                  | 
        in_storage: Storage,                                                                   | 
        in_shape: Shape,                                                                       | 
        in_strides: Strides,                                                                   | 
    ) -> None:                                                                                 | 
        if np.array_equal(out_shape, in_shape) and np.array_equal(out_strides, in_strides):    | 
            for i in prange(len(out)):---------------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                     | 
        else:                                                                                  | 
            for i in prange(len(out)):---------------------------------------------------------| #3
                out_index = np.zeros(MAX_DIMS, np.int32)---------------------------------------| #0
                in_index = np.zeros(MAX_DIMS, np.int32)----------------------------------------| #1
                to_index(i, out_shape, out_index)                                              | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                      | 
                o = index_to_position(out_index, out_strides)                                  | 
                j = index_to_position(in_index, in_strides)                                    | 
                out[o] = fn(in_storage[j])                                                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

# Zip
Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/kd/Library/Mobile Documents/com~apple~CloudDocs/Cornell Tech/2024Fall/MLE/mod3-kekelilii/minitorch/fast_ops.py (209) 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                                                  | 
        out: Storage,                                                                                                                                                          | 
        out_shape: Shape,                                                                                                                                                      | 
        out_strides: Strides,                                                                                                                                                  | 
        a_storage: Storage,                                                                                                                                                    | 
        a_shape: Shape,                                                                                                                                                        | 
        a_strides: Strides,                                                                                                                                                    | 
        b_storage: Storage,                                                                                                                                                    | 
        b_shape: Shape,                                                                                                                                                        | 
        b_strides: Strides,                                                                                                                                                    | 
    ) -> None:                                                                                                                                                                 | 
        if np.array_equal(out_shape, a_shape) and np.array_equal(out_strides, a_strides) and np.array_equal(out_shape, b_shape) and np.array_equal(out_strides, b_strides):    | 
            for i in prange(len(out)):-----------------------------------------------------------------------------------------------------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                                                                                                                        | 
        else:                                                                                                                                                                  | 
            out_index = np.zeros(MAX_DIMS, np.int32)---------------------------------------------------------------------------------------------------------------------------| #4
            a_index = np.zeros(MAX_DIMS, np.int32)-----------------------------------------------------------------------------------------------------------------------------| #5
            b_index = np.zeros(MAX_DIMS, np.int32)-----------------------------------------------------------------------------------------------------------------------------| #6
            for i in prange(len(out)):-----------------------------------------------------------------------------------------------------------------------------------------| #8
                to_index(i, out_shape, out_index)                                                                                                                              | 
                o = index_to_position(out_index, out_strides)                                                                                                                  | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                        | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                        | 
                aj = index_to_position(a_index, a_strides)                                                                                                                     | 
                bj = index_to_position(b_index, b_strides)                                                                                                                     | 
                out[o] = fn(a_storage[aj], b_storage[bj])                                                                                                                      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #4, #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
+--5 (parallel)
+--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--4 (parallel, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #4) had 2 loop(s) fused.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

# Reduce
Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/kd/Library/Mobile Documents/com~apple~CloudDocs/Cornell Tech/2024Fall/MLE/mod3-kekelilii/minitorch/fast_ops.py (260) 
---------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                 | 
        out: Storage,                                                            | 
        out_shape: Shape,                                                        | 
        out_strides: Strides,                                                    | 
        a_storage: Storage,                                                      | 
        a_shape: Shape,                                                          | 
        a_strides: Strides,                                                      | 
        reduce_dim: int,                                                         | 
    ) -> None:                                                                   | 
        reduce_size = a_shape[reduce_dim]                                        | 
                                                                                 | 
        for ordinal in prange(out.size):-----------------------------------------| #9
            out_index: Index = np.array([0] * len(out_shape), dtype=np.int32)    | 
            to_index(ordinal, out_shape, out_index)                              | 
                                                                                 | 
            a_index = out_index.copy()                                           | 
            a_index[reduce_dim] = 0                                              | 
            total = a_storage[index_to_position(a_index, a_strides)]             | 
                                                                                 | 
            for i in range(1, reduce_size):                                      | 
                a_index[reduce_dim] = i                                          | 
                a_ordinal = index_to_position(a_index, a_strides)                | 
                total = fn(total, a_storage[a_ordinal])                          | 
                                                                                 | 
            out[ordinal] = total                                                 | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

## Task 3.2 
# Matrix Multiply
Parallel loop listing for  Function _tensor_matrix_multiply, /Users/kd/Library/Mobile Documents/com~apple~CloudDocs/Cornell Tech/2024Fall/MLE/mod3-kekelilii/minitorch/fast_ops.py (289) 
-------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                             | 
    out: Storage,                                                        | 
    out_shape: Shape,                                                    | 
    out_strides: Strides,                                                | 
    a_storage: Storage,                                                  | 
    a_shape: Shape,                                                      | 
    a_strides: Strides,                                                  | 
    b_storage: Storage,                                                  | 
    b_shape: Shape,                                                      | 
    b_strides: Strides,                                                  | 
) -> None:                                                               | 
    """NUMBA tensor matrix multiply function.                            | 
                                                                         | 
    Should work for any tensor shapes that broadcast as long as          | 
                                                                         | 
    ```                                                                  | 
    assert a_shape[-1] == b_shape[-2]                                    | 
    ```                                                                  | 
                                                                         | 
    Optimizations:                                                       | 
                                                                         | 
    * Outer loop in parallel                                             | 
    * No index buffers or function calls                                 | 
    * Inner loop should have no global writes, 1 multiply.               | 
                                                                         | 
                                                                         | 
    Args:                                                                | 
    ----                                                                 | 
        out (Storage): storage for `out` tensor                          | 
        out_shape (Shape): shape for `out` tensor                        | 
        out_strides (Strides): strides for `out` tensor                  | 
        a_storage (Storage): storage for `a` tensor                      | 
        a_shape (Shape): shape for `a` tensor                            | 
        a_strides (Strides): strides for `a` tensor                      | 
        b_storage (Storage): storage for `b` tensor                      | 
        b_shape (Shape): shape for `b` tensor                            | 
        b_strides (Strides): strides for `b` tensor                      | 
                                                                         | 
    Returns:                                                             | 
    -------                                                              | 
        None : Fills in `out`                                            | 
                                                                         | 
    """                                                                  | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0               | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0               | 
                                                                         | 
    assert a_shape[-1] == b_shape[-2]                                    | 
                                                                         | 
    for i in prange(len(out)):-------------------------------------------| #13
        out_index: Index = np.zeros(len(out_shape), dtype=np.int32)------| #10
        tmp_i = i + 0                                                    | 
        to_index(tmp_i, out_shape, out_index)                            | 
        inner_loop = a_shape[-1]                                         | 
        for j in range(inner_loop):                                      | 
            tmp_j = j + 0                                                | 
            a_index: Index = np.zeros(len(a_shape), dtype=np.int32)------| #11
            b_index: Index = np.zeros(len(b_shape), dtype=np.int32)------| #12
            a_big_index = out_index.copy()                               | 
            b_big_index = out_index.copy()                               | 
            a_big_index[-1] = tmp_j                                      | 
            b_big_index[-2] = tmp_j                                      | 
            broadcast_index(a_big_index, out_shape, a_shape, a_index)    | 
            broadcast_index(b_big_index, out_shape, b_shape, b_index)    | 
            a_position = index_to_position(a_index, a_strides)           | 
            b_position = index_to_position(b_index, b_strides)           | 
            out[i] += a_storage[a_position] * b_storage[b_position]      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #10).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--13 is a parallel loop
   +--10 --> rewritten as a serial loop
   +--11 --> rewritten as a serial loop
   +--12 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--10 (parallel)
   +--11 (parallel)
   +--12 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--10 (serial)
   +--11 (serial)
   +--12 (serial)


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 3 loop(s) serialized as 
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
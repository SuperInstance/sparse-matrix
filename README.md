# Sparse Matrix

A Rust implementation of **sparse matrix** operations using a hash-map-based COO (Coordinate) format — storing only non-zero entries to achieve O(nnz) space for matrices with low fill ratios.

## Why It Matters

Real-world matrices are often extremely sparse. A social network graph might be a 10⁶×10⁶ matrix with only ~100 non-zeros per row (0.01% density). Storing all 10¹² entries densely would require terabytes; storing only the non-zeros needs megabytes. Sparse matrices are fundamental to scientific computing (finite element analysis), graph algorithms (adjacency matrices), machine learning (TF-IDF, recommender systems), and network analysis (PageRank). This implementation uses a HashMap keyed by `(row, col)` — not the most cache-efficient format, but the simplest to implement and understand.

## How It Works

### COO (Coordinate) Format via HashMap

Each non-zero entry is stored as a `(row, col) → value` pair in a `HashMap`:

```
SparseMatrix {
    rows: usize,
    cols: usize,
    data: HashMap<(usize, usize), f64>,
}
```

Only non-zero entries are stored. Setting a value to zero **removes** it from the map. This means `get(i, j)` for a zero entry is a HashMap miss that returns 0.0 — O(1) amortized.

### Matrix Multiplication

For `C = A × B`, iterate over A's non-zeros and for each `A[i,k]`, multiply by every `B[k,j]`:

```
for (i, k), a_ik in A.data:
    for j in 0..B.cols:
        b_kj = B.get(k, j)
        if b_kj ≠ 0:
            C[i,j] += a_ik × b_kj
```

Complexity: O(nnz(A) × B.cols) in the worst case. If B is also sparse, this is effectively O(nnz(A) × avg_nnz_per_row(B)).

### Matrix-Vector Multiplication

```
for (r, c), val in data:
    result[r] += val × v[c]
```

O(nnz) — the optimal complexity for sparse matvec. This is the operation in PageRank and conjugate gradient.

### Transpose

Swap keys: `(r, c) → (c, r)`. O(nnz).

### Addition

Merge entries: `C[i,j] = A[i,j] + B[i,j]`. O(nnz_A + nnz_B).

### Complexity

| Operation | Time | Notes |
|-----------|------|-------|
| `get(i, j)` | O(1) | HashMap lookup |
| `set(i, j, v)` | O(1) | HashMap insert/remove |
| `transpose()` | O(nnz) | Swap keys |
| `add(B)` | O(nnz_A + nnz_B) | Merge |
| `mul(B)` | O(nnz_A × cols_B) | Sparse-sparse matmul |
| `vector_mul(v)` | O(nnz) | Sparse matvec |
| Space | O(nnz) | Only non-zeros stored |

### Comparison to Other Formats

| Format | SpMV | SpGEMM | Insert | Notes |
|--------|------|--------|--------|-------|
| COO (HashMap) | O(nnz) | O(nnz_A × cols) | O(1) | This crate |
| CSR | O(nnz) | O(nnz_A + nnz_B) | O(nnz) | Best for read-heavy |
| CSC | O(nnz) | O(nnz_A + nnz_B) | O(nnz) | Column-oriented |
| DOK (Dict) | O(nnz) | Slow | O(1) | Build phase |
| LIL | O(nnz) | Slow | O(1) | Incremental build |

## Quick Start

```rust
fn main() {
    let mut a = SparseMatrix::new(4, 4);
    a.set(0, 0, 1.0);
    a.set(1, 1, 2.0);
    a.set(0, 2, 5.0);
    a.set(2, 0, -1.0);

    println!("{}×{} matrix, {} non-zeros ({:.1}% dense)",
        a.rows, a.cols, a.nnz(), a.density() * 100.0);

    // Matrix-vector multiply
    let v = vec![1.0, 1.0, 1.0, 1.0];
    let av = a.vector_mul(&v);
    println!("A·v = {:?}", av);

    // Transpose
    let at = a.transpose();
    // Add: A + A^T
    let symmetric = a.add(&at);
}
```

## API

### `SparseMatrix`
- `new(rows, cols)` — create empty (all-zero) matrix
- `set(r, c, val)` — set entry (removes if near-zero)
- `get(r, c) -> f64` — get entry (0.0 if absent)
- `nnz() -> usize` — number of non-zeros
- `density() -> f64` — fraction of non-zero entries
- `transpose() -> SparseMatrix`
- `add(other) -> SparseMatrix`
- `mul(other) -> SparseMatrix`
- `vector_mul(v) -> Vec<f64>`
- `print_dense()` — debug output

## Architecture Notes

In SuperInstance, sparse matrices represent the fleet ship connectivity graph — mostly sparse since each ship communicates with a small subset of peers. Matrix-vector products propagate conservation state (γ, η values) through the fleet, and the γ + η = C law is checked at each step. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Davis, T. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM.
- Buluç, A. & Gilbert, J. (2012). *Parallel Sparse Matrix-Matrix Multiplication and Indexing*. Encyclopedia of Parallel Computing.

## License

MIT

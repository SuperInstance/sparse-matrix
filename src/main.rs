use std::collections::HashMap;

#[derive(Clone, Debug)]
struct SparseMatrix {
    rows: usize,
    cols: usize,
    data: HashMap<(usize, usize), f64>,
}

impl SparseMatrix {
    fn new(rows: usize, cols: usize) -> Self {
        SparseMatrix {
            rows,
            cols,
            data: HashMap::new(),
        }
    }

    fn set(&mut self, r: usize, c: usize, val: f64) {
        assert!(r < self.rows && c < self.cols, "Index out of bounds");
        if val.abs() > 1e-15 {
            self.data.insert((r, c), val);
        } else {
            self.data.remove(&(r, c));
        }
    }

    fn get(&self, r: usize, c: usize) -> f64 {
        *self.data.get(&(r, c)).unwrap_or(&0.0)
    }

    fn nnz(&self) -> usize {
        self.data.len()
    }

    fn density(&self) -> f64 {
        self.nnz() as f64 / (self.rows * self.cols) as f64
    }

    fn transpose(&self) -> SparseMatrix {
        let mut result = SparseMatrix::new(self.cols, self.rows);
        for (&(r, c), &v) in &self.data {
            result.data.insert((c, r), v);
        }
        result
    }

    fn add(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = self.clone();
        for (&(r, c), &v) in &other.data {
            let new_val = result.get(r, c) + v;
            result.set(r, c, new_val);
        }
        result
    }

    fn mul(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = SparseMatrix::new(self.rows, other.cols);
        for (&(i, k), &a_ik) in &self.data {
            for j in 0..other.cols {
                let b_kj = other.get(k, j);
                if b_kj.abs() > 1e-15 {
                    let cur = result.get(i, j);
                    result.set(i, j, cur + a_ik * b_kj);
                }
            }
        }
        result
    }

    fn vector_mul(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        let mut result = vec![0.0; self.rows];
        for (&(r, c), &val) in &self.data {
            result[r] += val * v[c];
        }
        result
    }

    fn print_dense(&self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                print!("{:8.2}", self.get(i, j));
            }
            println!();
        }
    }
}

fn main() {
    let mut a = SparseMatrix::new(4, 4);
    a.set(0, 0, 1.0);
    a.set(1, 1, 2.0);
    a.set(2, 2, 3.0);
    a.set(3, 3, 4.0);
    a.set(0, 2, 5.0);
    a.set(2, 0, -1.0);

    println!("Matrix A ({}x{}, {} nnz, {:.1}% dense):", a.rows, a.cols, a.nnz(), a.density() * 100.0);
    a.print_dense();

    println!("\nA^T:");
    let at = a.transpose();
    at.print_dense();

    let v = vec![1.0, 1.0, 1.0, 1.0];
    let av = a.vector_mul(&v);
    println!("\nA * [1,1,1,1] = {:?}", av);

    let c = a.add(&at);
    println!("\nA + A^T ({} nnz):", c.nnz());
    c.print_dense();
}

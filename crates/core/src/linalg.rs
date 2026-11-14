//! Tiny fixed-size linear algebra: just what the solvers need.
//! Matrices are row-major `[[f64; 3]; 3]`.

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

pub const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn normalize(a: V3) -> V3 {
    let n = norm(a);
    if n < 1e-300 {
        [0.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / n)
    }
}
pub fn mid(a: V3, b: V3) -> V3 {
    scale(add(a, b), 0.5)
}
/// Angle between two vectors, radians.
pub fn angle(a: V3, b: V3) -> f64 {
    let c = dot(normalize(a), normalize(b)).clamp(-1.0, 1.0);
    c.acos()
}
/// Component of `a` orthogonal to unit vector `u`.
pub fn reject(a: V3, u: V3) -> V3 {
    sub(a, scale(u, dot(a, u)))
}

pub fn mat_vec(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
pub fn mat_mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}
pub fn transpose(m: &M3) -> M3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}
/// Matrix whose columns are a, b, c.
pub fn from_cols(a: V3, b: V3, c: V3) -> M3 {
    [[a[0], b[0], c[0]], [a[1], b[1], c[1]], [a[2], b[2], c[2]]]
}

/// Rodrigues: rotation vector -> rotation matrix.
pub fn rodrigues(w: V3) -> M3 {
    let th = norm(w);
    if th < 1e-12 {
        // first-order
        return [[1.0, -w[2], w[1]], [w[2], 1.0, -w[0]], [-w[1], w[0], 1.0]];
    }
    let k = scale(w, 1.0 / th);
    let (s, c) = th.sin_cos();
    let v = 1.0 - c;
    [
        [c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s],
        [k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s],
        [k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v],
    ]
}

/// Smallest rotation taking unit vector `a` onto unit vector `b`.
pub fn rot_between(a: V3, b: V3) -> M3 {
    let a = normalize(a);
    let b = normalize(b);
    let axis = cross(a, b);
    let s = norm(axis);
    let c = dot(a, b);
    if s < 1e-12 {
        if c > 0.0 {
            return I3;
        }
        // 180°: any perpendicular axis
        let p = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let ax = normalize(cross(a, p));
        return rodrigues(scale(ax, std::f64::consts::PI));
    }
    rodrigues(scale(axis, s.atan2(c) / s))
}

/// Solve a small dense linear system A x = b (n ≤ 8) by Gaussian elimination
/// with partial pivoting. `a` is row-major n×n. Returns None if singular.
pub fn solve_dense<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let mut piv = col;
        for r in col + 1..N {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..N {
            let f = a[r][col] / a[col][col];
            if f != 0.0 {
                for c in col..N {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = [0.0; N];
    for i in (0..N).rev() {
        let mut s = b[i];
        for j in i + 1..N {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rodrigues_orthonormal() {
        let r = rodrigues([0.3, -0.7, 1.1]);
        let rt = mat_mul(&r, &transpose(&r));
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((rt[i][j] - e).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn rot_between_maps() {
        let a = normalize([0.2, 0.5, 1.0]);
        let b = normalize([-0.4, 0.1, 0.8]);
        let r = rot_between(a, b);
        let m = mat_vec(&r, a);
        assert!(norm(sub(m, b)) < 1e-12);
    }
    #[test]
    fn dense_solve() {
        let x = solve_dense([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]], [3.0, 5.0, 5.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12 && (x[2] - 1.0).abs() < 1e-12);
    }
}

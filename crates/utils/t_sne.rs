use rand_distr::{Normal, Distribution};

// DS
//QwQ + DS_main
fn compute_squared_distances(data: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = data.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut dist = 0.0;
            for k in 0..data[i].len() {
                let diff = data[i][k] - data[j][k];
                dist += diff * diff;
            }
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }
    distances
}

fn find_sigma(distances_i: &[f32], target_entropy: f32, max_iter: usize) -> f32 {
    let mut lower = 1e-20;
    let mut upper = 1e20;
    let mut best_sigma = 1.0;
    let mut min_error = f32::MAX;

    for _ in 0..max_iter {
        let sigma:f32 = (lower + upper) / 2.0;
        let mut sum_p = 0.0;
        let mut entropy = 0.0;
        for &dist in distances_i {
            let exp_val = (-dist / (2.0 * sigma.powi(2))).exp();
            sum_p += exp_val;
        }
        let sum_p_recip = 1.0 / sum_p;
        for &dist in distances_i {
            let exp_val = (-dist / (2.0 * sigma.powi(2))).exp();
            let p = exp_val * sum_p_recip;
            entropy += p * p.ln();
        }
        entropy = -entropy;

        let error = (entropy - target_entropy).abs();
        if error < min_error {
            min_error = error;
            best_sigma = sigma;
        }

        if entropy > target_entropy {
            upper = sigma;
        } else {
            lower = sigma;
        }
    }
    best_sigma
}

fn compute_p_matrix(data: &Vec<Vec<f32>>, perplexity: f32) -> Vec<Vec<f32>> {
    let n = data.len();
    let distances = compute_squared_distances(data);
    let target_entropy = perplexity.ln();
    let mut p_joint = vec![vec![0.0; n]; n];

    for i in 0..n {
        let distances_i: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| distances[i][j]).collect();
        let sigma_i = find_sigma(&distances_i, target_entropy, 200);

        let mut sum_p = 0.0;
        for j in 0..n {
            if i != j {
                let dist = distances[i][j];
                let exp_val = (-dist / (2.0 * sigma_i.powi(2))).exp();
                sum_p += exp_val;
            }
        }

        let sum_p_recip = 1.0 / sum_p;
        for j in 0..n {
            if i != j {
                let dist = distances[i][j];
                let exp_val = (-dist / (2.0 * sigma_i.powi(2))).exp();
                p_joint[i][j] = exp_val * sum_p_recip;
            } else {
                p_joint[i][j] = 0.0;
            }
        }
    }

    let mut p_sym = vec![vec![0.0; n]; n];
    let n_f32 = n as f32;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                p_sym[i][j] = (p_joint[i][j] + p_joint[j][i]) / (2.0 * n_f32);
            }
        }
    }

    let epsilon = 1e-12;
    let mut sum_p_sym = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                p_sym[i][j] += epsilon;
                sum_p_sym += p_sym[i][j];
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            if i != j {
                p_sym[i][j] /= sum_p_sym;
            }
        }
    }

    p_sym
}

fn initialize_y(n: usize, n_components: usize) -> Vec<Vec<f32>> {
    let normal = Normal::new(0.0, 1e-4).unwrap();
    let mut rng = rand::rng();
    (0..n)
        .map(|_| (0..n_components).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

fn run_tsne(
    p_sym: &Vec<Vec<f32>>,
    y: &mut Vec<Vec<f32>>,
    n_components: usize,
    learning_rate: f32,
    momentum: f32,
    num_iterations: usize,
) {
    let n = y.len();
    let mut vel_y = vec![vec![0.0; n_components]; n];

    for _ in 0..num_iterations {
        let mut y_dist = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut dist = 0.0;
                for k in 0..n_components {
                    let diff = y[i][k] - y[j][k];
                    dist += diff * diff;
                }
                y_dist[i][j] = dist;
            }
        }

        let mut q = vec![vec![0.0; n]; n];
        let mut sum_q = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    q[i][j] = 1.0 / (1.0 + y_dist[i][j]);
                    sum_q += q[i][j];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    q[i][j] /= sum_q;
                }
            }
        }

        let mut gradient = vec![vec![0.0; n_components]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let p = p_sym[i][j];
                let q_val = q[i][j];
                let factor = (p - q_val) / (1.0 + y_dist[i][j]);
                for k in 0..n_components {
                    gradient[i][k] += 4.0 * factor * (y[i][k] - y[j][k]);
                }
            }
        }

        for i in 0..n {
            for k in 0..n_components {
                vel_y[i][k] = momentum * vel_y[i][k] - learning_rate * gradient[i][k];
                y[i][k] += vel_y[i][k];
            }
        }
    }
}

pub fn tsne(
    data: &Vec<Vec<f32>>,
    n_components: usize,
    perplexity: f32,
    learning_rate: f32,
    momentum: f32,
    num_iterations: usize,
) -> Vec<Vec<f32>> {
    let p_sym = compute_p_matrix(data, perplexity);
    let mut y = initialize_y(data.len(), n_components);
    run_tsne(&p_sym, &mut y, n_components, learning_rate, momentum, num_iterations);
    y
}
#[cfg(test)]
mod tests {
    use super::*;
    // Example usage
    #[test]
    fn test_tsne() {
        // Example data: 2D vectors
        let data: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
        ];

        let perplexity = 30.0;
        let iterations = 1000;
        let learning_rate = 200.0;
        let momentum = 0.1;

        let embedded = tsne(&data, 2, perplexity, learning_rate, momentum ,iterations);
        println!("{:?}", embedded);
    }
}
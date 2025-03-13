use rand::Rng;
use rand::rngs::ThreadRng;

// qwq32b
pub fn umap(
    data: &[Vec<f32>],
    n_neighbors: usize,
    n_components: usize,
    learning_rate: f32,
    n_epochs: usize,
    min_dist: f32,
    //spread: f32,
    metric: &str,
) -> Result<Vec<Vec<f32>>, String> {
    let n = data.len();
    let d = data[0].len();

    // Check all data points have the same dimensionality
    for row in data.iter() {
        if row.len() != d {
            return Err("All data points must have the same dimensionality".to_string());
        }
    }

    // Compute pairwise distances
    let distance_matrix = compute_distance_matrix(data, metric)?;

    // Compute epsilon for each point based on k-nearest neighbors
    let epsilons = compute_epsilons(&distance_matrix, n_neighbors);

    // Compute global epsilon (median of epsilons)
    let global_epsilon = compute_global_epsilon(&epsilons);

    // Compute local and global similarities
    let local_similarities = compute_local_similarities(&distance_matrix, &epsilons);
    let global_similarities = compute_global_similarities(&distance_matrix, global_epsilon);

    // Combine similarities to form the fuzzy set
    let combined_similarities = combine_similarities(&local_similarities, &global_similarities);

    // Normalize to get the high-dimensional probabilities p_ij
    let p_matrix = normalize_probabilities(&combined_similarities);

    // Initialize low-dimensional embeddings randomly
    let mut embeddings = initialize_embeddings(n, n_components);

    // Optimization loop
    for _ in 0..n_epochs {
        // Compute gradients and update embeddings
        embeddings = optimize_embeddings(
            &embeddings,
            &p_matrix,
            min_dist,
            learning_rate,
            //&distance_matrix,
        );
    }

    Ok(embeddings)
}

// Helper functions
fn compute_distance_matrix(data: &[Vec<f32>], metric: &str) -> Result<Vec<Vec<f32>>, String> {
    let n = data.len();
    let mut distance_matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                distance_matrix[i][j] = 0.0;
            } else {
                let dist = match metric {
                    "euclidean" => euclidean_distance(&data[i], &data[j]),
                    "cosine" => cosine_distance(&data[i], &data[j]),
                    _ => return Err("Unsupported metric".to_string()),
                };
                distance_matrix[i][j] = dist;
            }
        }
    }
    Ok(distance_matrix)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    1.0 - dot_product / (norm_a * norm_b)
}

fn compute_epsilons(distance_matrix: &[Vec<f32>], n_neighbors: usize) -> Vec<f32> {
    let n = distance_matrix.len();
    let mut epsilons = vec![0.0; n];

    for i in 0..n {
        let mut indices: Vec<_> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            distance_matrix[i][a]
                .partial_cmp(&distance_matrix[i][b])
                .unwrap()
        });
        indices.retain(|&j| j != i); // Exclude self
        let epsilon_i = distance_matrix[i][indices[n_neighbors - 1]];
        epsilons[i] = epsilon_i;
    }

    epsilons
}

fn compute_global_epsilon(epsilons: &[f32]) -> f32 {
    let mut sorted = epsilons.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    sorted[mid]
}


fn combine_similarities(
    local: &[Vec<f32>],
    global: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let n = local.len();
    let mut combined = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            combined[i][j] = local[i][j].min(global[i][j]);
        }
    }

    combined
}
fn compute_local_similarities(
    distance_matrix: &[Vec<f32>],
    epsilons: &[f32],
) -> Vec<Vec<f32>> {
    let n = distance_matrix.len();
    let mut similarities = vec![vec![0.0; n]; n];

    for i in 0..n {
        let epsilon_i = epsilons[i];
        for j in 0..n {
            if i == j {
                similarities[i][j] = 0.0;
            } else {
                let d = distance_matrix[i][j];
                let exponent = -d.powi(2) / (2.0 * epsilon_i.powi(2));
                similarities[i][j] = exponent.exp();
            }
        }
    }

    similarities
}

fn compute_global_similarities(
    distance_matrix: &[Vec<f32>],
    global_epsilon: f32,
) -> Vec<Vec<f32>> {
    let n = distance_matrix.len();
    let mut similarities = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                similarities[i][j] = 0.0;
            } else {
                let d = distance_matrix[i][j];
                let exponent = -d.powi(2) / (2.0 * global_epsilon.powi(2));
                similarities[i][j] = exponent.exp();
            }
        }
    }

    similarities
}
fn normalize_probabilities(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = matrix.len();
    let total = matrix.iter().map(|row| row.iter().sum::<f32>()).sum::<f32>();
    let mut p_matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            p_matrix[i][j] = matrix[i][j] / total;
        }
    }

    p_matrix
}

fn initialize_embeddings(n: usize, n_components: usize) -> Vec<Vec<f32>> {
    let mut rng:ThreadRng = rand::rng();
    (0..n)
        .map(|_| {
            (0..n_components)
                .map(|_| (rng.random::<f32>() - 0.5) * 1e-4)
                .collect()
        })
        .collect()
}

fn optimize_embeddings(
    embeddings: &[Vec<f32>],
    p_matrix: &[Vec<f32>],
    min_dist: f32,
    learning_rate: f32,
) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let n_components = embeddings[0].len();
    let mut gradients = vec![vec![0.0; n_components]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d_low = euclidean_distance(&embeddings[i], &embeddings[j]);
            let q_ij = 1.0 / (1.0 + (d_low / (1.0 - min_dist)).powi(2));
            let delta = p_matrix[i][j] - q_ij;
            let direction = embeddings[i]
                .iter()
                .zip(embeddings[j].iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();

            let grad_part = 4.0 * delta * (1.0 / (1.0 + d_low.powi(2)));

            for (c, dir) in direction.iter().enumerate().take(n_components) {
                gradients[i][c] += grad_part * dir;
                gradients[j][c] -= grad_part * dir;
            }
        }
    }

    // Update embeddings
    embeddings
        .iter()
        .zip(gradients.iter())
        .map(|(emb, grad)| {
            emb.iter()
                .zip(grad.iter())
                .map(|(&e, &g)| e + learning_rate * g)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    // Example usage
    #[test]
    fn test_umap() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];

        let result = umap(
            &data,
            2,
            2,
            0.1,
            100,
            0.1,
            // 1.0,
            "euclidean",
        ).unwrap();

        for point in result {
            println!("{:?}", point);
        }
    }
}
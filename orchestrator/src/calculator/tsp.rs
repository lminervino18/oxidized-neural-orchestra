use std::ops::Add;

use super::Graph;

/// Calculates the best cycle through the given graph.
///
/// The amount of vertices is capped at `usize::BITS`. If a larger graph is
/// given the result will be an empty `Vec`.
///
/// # Args
/// * `graph` - The graph containing all the weights.
///
/// # Returns
/// The best cycle optimized for cost in the graph.
pub fn travelling_salesman_cycle<W>(graph: &Graph<W>) -> Vec<usize>
where
    W: Copy + Add<Output = W> + PartialOrd + Default,
{
    if graph.is_empty() {
        return Vec::new();
    }

    if graph.len() > usize::BITS as usize {
        return Vec::new();
    }

    let n = graph.len();
    let exec_tree_size = n * (1 << n);
    let mut mem = vec![None; exec_tree_size];
    let mut parent = vec![None; exec_tree_size];

    if dp(graph, 1, 0, &mut mem, &mut parent).is_none() {
        return Vec::new();
    }

    reconstruct_solution(n, &parent)
}

/// Navigates through the different permutations of vertices in the graph avoiding recomputation of
/// state nodes for already traced subpaths.
///
/// # Args
/// * `graph` - The graph containing all the weights.
/// * `visited` - A bitmask of the visited nodes where the `w` vertex is represented by the `1 << w`'th bit being set.
/// * `v` - The last vertex of the current path.
/// * `mem` - A memoization table for previous calculations.
/// * `parent` - The parent table to reconstruct a potential optimal solution later.
///
/// # Returns
/// `Some(cost)` if a cycle exists, `None` otherwise.
fn dp<W>(
    graph: &Graph<W>,
    visited: usize,
    v: usize,
    mem: &mut Vec<Option<W>>,
    parent: &mut Vec<Option<usize>>,
) -> Option<W>
where
    W: Copy + PartialOrd + Add<Output = W> + Default,
{
    let n = graph.len();
    let i = visited * n + v;

    if let Some(solution) = mem[i] {
        return Some(solution);
    }

    if visited == (1 << n) - 1 {
        return graph.get_weight(v, 0);
    }

    for w in 1..n {
        let bit = 1 << w;

        if visited & bit == 0
            && let Some(weight) = graph.get_weight(v, w)
            && let Some(cost) = dp(graph, visited | bit, w, mem, parent)
        {
            let cycle_cost = weight + cost;

            if mem[i].is_none_or(|min| cycle_cost < min) {
                mem[i] = Some(cycle_cost);
                parent[i] = Some(w);
            }
        }
    }

    mem[i]
}

/// Navigates through the parent table building the cycle found by the `dp` function.
///
/// # Args
/// * `n` - The size of the graph.
/// * `parent` - The parent table indicating the path from one vertex to another.
///
/// # Returns
/// The best hamiltonian cycle starting from vertex `0`.
fn reconstruct_solution(n: usize, parent: &[Option<usize>]) -> Vec<usize> {
    let mut path = Vec::with_capacity(n);
    let mut visited = 1;
    let mut v = 0;

    path.push(v);

    while path.len() < n {
        let i = visited * n + v;

        // SAFETY: If reconstruct_solution is invoqued, it means dp found
        //         a hamiltonian cycle, meaning that this path should be
        //         entirely mapped onto the parent table.
        let w = parent[i].unwrap();
        visited |= 1 << w;
        path.push(w);
        v = w;
    }

    path
}

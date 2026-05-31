use std::{collections::HashSet, num::NonZeroUsize};

use super::Graph;

/// Calculates the `k` vertices that are the most centered in a `K_{k, n - k}` subgraph of `graph`.
///
/// The amount of vertices is capped at `usize::BITS`. If a larger graph is
/// inputed the result will be an empty `Vec`.
///
/// # Args
/// * `graph` - The graph containing all the weights.
/// * `k` - The amount of central nodes.
///
/// # Returns
/// The set of the k most central vertices.
pub fn central_vertices<W>(graph: &Graph<W>, k: usize) -> HashSet<usize>
where
    W: Copy + Ord,
{
    if graph.is_empty() {
        return HashSet::new();
    }

    let n = graph.len();

    if n > usize::BITS as usize {
        return HashSet::new();
    }

    if k >= n {
        return (0..n).collect();
    }

    let Some(k) = NonZeroUsize::new(k) else {
        return HashSet::new();
    };

    let center_bitmask = bt(graph, k, 0, 0, 0, None).0;
    let mut center = HashSet::with_capacity(k.get());

    for i in 0..usize::BITS as usize {
        if center_bitmask & (1 << i) != 0 {
            center.insert(i);
        }

        if center.len() == k.get() {
            break;
        }
    }

    center
}

/// Backtracking exploration through every central vertices combination.
///
/// # Args
/// * `graph` - The graph containing all the weights.
/// * `k` - The amount of central vertices to obtain.
/// * `start` - The starting search index.
/// * `center` - The current central vertices.
/// * `opt` - The best solution found.
/// * `opt_min` - The max weight in the optimal solution found.
///
/// # Returns
/// The optimal solution and it's max weight.
fn bt<W>(
    graph: &Graph<W>,
    k: NonZeroUsize,
    start: usize,
    center: usize,
    mut opt: usize,
    mut opt_min: Option<W>,
) -> (usize, Option<W>)
where
    W: Copy + Ord,
{
    let in_center = center.count_ones() as usize;

    if in_center == k.get() {
        // SAFETY: The length of the central vertices slice is positive.
        let score = eval_centrality(graph, center).unwrap();

        if opt_min.is_none_or(|min| score < min) {
            opt = center;
            opt_min = Some(score);
        }

        return (opt, opt_min);
    }

    let n = graph.len();
    let left_to_find = k.get() - in_center;
    let left_to_search = n - start;

    if left_to_find > left_to_search {
        return (opt, opt_min);
    }

    for i in start..n {
        let bit = 1 << i;
        (opt, opt_min) = bt(graph, k, i + 1, center | bit, opt, opt_min);
    }

    (opt, opt_min)
}

/// Evaluates the current centrality of the vertices.
///
/// # Args
/// * `graph` - The graph containing all the weights.
/// * `center` - The current central vertices.
///
/// # Returns
/// The maximum weight of centrality or `None` if there are no central nodes.
fn eval_centrality<W>(graph: &Graph<W>, center: usize) -> Option<W>
where
    W: Copy + Ord,
{
    let n = graph.len();
    let mut max = None;

    for v in 0..n {
        if (center & (1 << v)) != 0 {
            continue;
        }

        for w in 0..n {
            if (center & (1 << w)) == 0 {
                continue;
            }

            if let Some(weight) = graph.get_weight(v, w) {
                max = Some(max.unwrap_or(weight).max(weight));
            }
        }
    }

    max
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn assert_are_central<I>(center: HashSet<usize>, expected: I)
    where
        I: IntoIterator<Item = usize>,
    {
        let expected_set: HashSet<_> = expected.into_iter().collect();
        assert_eq!(center, expected_set);
    }

    #[test]
    fn empty_graph() {
        let graph: Graph<usize> = Graph::new(0, []).unwrap();
        let center = central_vertices(&graph, 1);
        assert_eq!(center, HashSet::<usize>::new());
    }

    #[test]
    fn null_k() {
        let edges = [
            (0, 1, 1), //
            (0, 2, 1), //
            (1, 2, 1), //
        ];

        let graph = Graph::new(3, edges).unwrap();
        let center = central_vertices(&graph, 0);

        assert_eq!(center, HashSet::<usize>::new());
    }

    #[test]
    fn small_test() {
        let edges = [
            (0, 1, 5), //
            (0, 2, 2), //
            (1, 2, 2), //
        ];

        let graph = Graph::new(3, edges).unwrap();
        let center = central_vertices(&graph, 2);

        assert_are_central(center, [0, 1]);
    }

    #[test]
    fn medium_test() {
        let edges = [
            (0, 1, 1),    //
            (0, 2, 1),    //
            (0, 3, 1),    //
            (1, 2, 1000), //
            (1, 3, 1000), //
            (2, 3, 1000), //
        ];

        let graph = Graph::new(4, edges).unwrap();
        let center = central_vertices(&graph, 3);

        assert_are_central(center, [1, 2, 3]);
    }

    #[test]
    fn large_test() {
        let edges = [
            (0, 1, 1000),
            (0, 2, 1),
            (0, 3, 1000),
            (0, 4, 1),
            (1, 2, 1),
            (1, 3, 1000),
            (1, 4, 1),
            (2, 3, 1),
            (2, 4, 1),
            (3, 4, 1),
        ];

        let graph = Graph::new(5, edges).unwrap();
        let center = central_vertices(&graph, 3);

        assert_are_central(center, [0, 1, 3]);
    }
}

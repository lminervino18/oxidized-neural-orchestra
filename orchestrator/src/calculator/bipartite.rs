use std::num::NonZeroUsize;

use super::Graph;

/// Calculates the `k` vertices that are the best centered in a `K_{k, n - k}` graph.
///
/// # Args
/// * `graph` - The graph containing all the weights.
/// * `k` - The amount of central nodes.
///
/// # Returns
/// The set of the k most central vertices.
pub fn bipartite_center<W>(graph: Graph<W>, k: usize) -> Vec<usize>
where
    W: Copy + Ord,
{
    let Some(k) = NonZeroUsize::new(k) else {
        return Vec::new();
    };

    if graph.is_empty() {
        return Vec::new();
    }

    let n = graph.len();

    if k.get() >= n {
        return (0..n).collect();
    }

    let mut center = Vec::with_capacity(k.get());
    let opt = Vec::with_capacity(k.get());

    bt(&graph, k, 0, &mut center, opt, None).0
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
    center: &mut Vec<usize>,
    mut opt: Vec<usize>,
    mut opt_score: Option<W>,
) -> (Vec<usize>, Option<W>)
where
    W: Copy + Ord,
{
    if center.len() == k.get() {
        // SAFETY: The length of the central vertices slice is positive.
        let score = eval_centrality(graph, center).unwrap();

        if opt_score.is_none_or(|min| score < min) {
            return (center.clone(), Some(score));
        }

        return (opt, opt_score);
    }

    let n = graph.len();
    let left_to_find = k.get() - center.len();
    let left_to_search = n - start;

    if left_to_find > left_to_search {
        return (opt, opt_score);
    }

    for i in start..n {
        center.push(i);
        (opt, opt_score) = bt(graph, k, i + 1, center, opt, opt_score);
        center.pop();
    }

    (opt, opt_score)
}

/// Evaluates the current centrality of the vertices.
///
/// # Args
/// * `graph` - The graph containing all the weights.
/// * `center` - The current central vertices.
///
/// # Returns
/// The maximum weight of centrality or `None` if there are no central nodes.
fn eval_centrality<W>(graph: &Graph<W>, center: &[usize]) -> Option<W>
where
    W: Copy + Ord,
{
    let n = graph.len();
    let periphery = (0..n).filter(|v| !center.contains(v));
    let mut max = None;

    for v in periphery {
        for &w in center {
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

    fn assert_are_central<I>(center: Vec<usize>, expected: I)
    where
        I: IntoIterator<Item = usize>,
    {
        let expected_set: HashSet<_> = expected.into_iter().collect();
        let center_set: HashSet<_> = center.into_iter().collect();
        assert_eq!(center_set, expected_set);
    }

    #[test]
    fn empty_graph() {
        let graph: Graph<usize> = Graph::new(0, []).unwrap();
        let center = bipartite_center(graph, 1);
        assert_eq!(center, Vec::<usize>::new());
    }

    #[test]
    fn null_k() {
        let edges = [
            (0, 1, 1), //
            (0, 2, 1), //
            (1, 2, 1), //
        ];

        let graph = Graph::new(3, edges).unwrap();
        let center = bipartite_center(graph, 0);

        assert_eq!(center, Vec::<usize>::new());
    }

    #[test]
    fn small_test() {
        let edges = [
            (0, 1, 5), //
            (0, 2, 2), //
            (1, 2, 2), //
        ];

        let graph = Graph::new(3, edges).unwrap();
        let center = bipartite_center(graph, 2);

        assert_are_central(center, [0, 1]);
    }

    #[test]
    fn medium_test() {
        let edges = [
            (0, 1, 1), //
            (0, 2, 1), //
            (0, 3, 1), //
        ];

        let graph = Graph::new(4, edges).unwrap();
        let center = bipartite_center(graph, 3);

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
        let center = bipartite_center(graph, 3);

        assert_are_central(center, [0, 1, 3]);
    }
}

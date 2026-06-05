/// A simple graph implementatino for calculating the best Ring All Reduce route between workers
/// and the best server distribution given any node topology for Parameter Server.
pub struct Graph<W>
where
    W: Copy,
{
    matrix: Vec<Option<W>>,
    size: usize,
}

impl<W> Graph<W>
where
    W: Copy,
{
    /// Creates a new `Graph`.
    ///
    /// # Args
    /// * `size` - The amount of vertices.
    /// * `edges` - The edges of the graph.
    ///
    /// # Returns
    /// Self if the edges are valid, or `None` if it's indices are out of bounds.
    pub fn new<I>(size: usize, edges: I) -> Option<Self>
    where
        I: IntoIterator<Item = (usize, usize, W)>,
    {
        let mut matrix = vec![None; size.pow(2)];

        for (i, j, weight) in edges {
            if i >= size || j >= size {
                return None;
            }

            matrix[i * size + j] = Some(weight);
            matrix[j * size + i] = Some(weight);
        }

        Some(Self { matrix, size })
    }

    /// Retrieves the weight from edge `(i, j)`.
    ///
    /// # Args
    /// * `i` - A vertex.
    /// * `j` - A vertex.
    ///
    /// # Returns
    /// `Some(T)` if the edge is inbounds, `None` otherwise.
    pub fn get_weight(&self, i: usize, j: usize) -> Option<W> {
        if i < self.size && j < self.size {
            self.matrix[i * self.size + j]
        } else {
            None
        }
    }

    /// Retrieves the size of the graph.
    ///
    /// # Returns
    /// The amount of vertices in the graph.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Tells if the graph doesn't have any vertices
    ///
    /// # Returns
    /// Either `true` if the graph is empty or `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

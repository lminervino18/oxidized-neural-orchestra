use std::{collections::HashSet, mem};

/// Manages the parameters for the trainer.
///
/// It will accumulate the parameters from different servers and yield
/// them according to the starting ordering provided in the constructor.
///
/// The parameters can be iterated sequentially in order or in reverse
/// through the `FrontIter` and the `BackIter`.
pub struct ParamManager<'rx> {
    ordering: Vec<usize>,
    params: Vec<Option<&'rx mut [f32]>>,
}

impl<'rx> ParamManager<'rx> {
    /// Creates a new `ParamManager`.
    ///
    /// # Arguments
    /// * `ordering` - The ordering for each server's parameters.
    ///
    /// # Returns
    /// A new `ParamManager` instance.
    pub fn new(ordering: Vec<usize>) -> Option<Self> {
        // TODO: Replace `Option` with a `Result` of some kind.
        if ordering.is_empty() {
            return None;
        }

        let mut max = ordering[0];
        let mut unique = HashSet::with_capacity(ordering.len());

        for &id in ordering.iter() {
            max = max.max(id);
            unique.insert(id);
        }

        if unique.len() != max + 1 {
            return None;
        }

        let manager = Self {
            ordering,
            params: (0..unique.len()).map(|_| None).collect(),
        };

        Some(manager)
    }

    /// Adds parameters to the inner parameter buffer for this server.
    ///
    /// # Arguments
    /// * `server_id` - The id of the server that sent the given parameters.
    /// * `params` - The parameters to hold on to.
    ///
    /// # Returns
    /// `None` if the requested id is out of bounds or the given id already had an associated parameter slice.
    pub fn add(&mut self, server_id: usize, params: &'rx mut [f32]) -> Option<()> {
        // TODO: Return an error if the requested id is out of bounds instead of panicing.
        // TODO: Return an error if there were parameters for this server.
        let server_params = self.params.get_mut(server_id)?;

        if server_params.replace(params).is_some() {
            return None;
        }

        Some(())
    }

    /// Clears the parameters the inner parameter list. This should be called
    /// just after sending the new gradients to the server.
    pub fn clear(&mut self) {
        for params in &mut self.params {
            params.take();
        }
    }

    /// Creates a new iterator for the server parameters.
    ///
    /// # Returns
    /// A new `FrontIter` iterator, or `None` if there are any missing parameters.
    pub fn front(&mut self) -> Option<FrontIter<'_>> {
        let mut slices = Vec::with_capacity(self.params.len());

        for param in &mut self.params {
            slices.push(&mut **param.as_mut()?);
        }

        let front = FrontIter {
            ordering: &self.ordering,
            slices,
            curr: 0,
        };

        Some(front)
    }

    /// Creates a new iterator for the server parameters.
    ///
    /// # Returns
    /// A new `BackIter` iterator, or `None` if there are any missing parameters.
    pub fn back(&mut self) -> Option<BackIter<'_>> {
        let mut slices = Vec::with_capacity(self.params.len());

        for param in &mut self.params {
            slices.push(&mut **param.as_mut()?);
        }

        let back = BackIter {
            ordering: &self.ordering,
            slices,
            curr: 0,
        };

        Some(back)
    }
}

/// The servers' parameter iterator.
pub struct FrontIter<'pm> {
    ordering: &'pm [usize],
    slices: Vec<&'pm mut [f32]>,
    curr: usize,
}

impl<'pm> FrontIter<'pm> {
    /// Takes the next batch of `n` parameters following the manager's ordering.
    ///
    /// # Arguments
    /// * `n` - The amount of parameters to take.
    ///
    /// # Returns
    /// A slice of parameters or `None` if the iteration has ended.
    pub fn take(&mut self, n: usize) -> Option<&'pm mut [f32]> {
        let server_id = *self.ordering.get(self.curr)?;
        self.curr += 1;

        // SAFETY: `ParamManager`'s constructor validates that every
        //         value in the ordering is lower than it's length.
        let sv_slice = mem::take(&mut self.slices[server_id]);

        if n > sv_slice.len() {
            // TODO: Ver como podemos garantizar la invariante, que no se pidan mas parametros
            //       de los que de verdad se tiene.
            //
            //      Si se entra aca es porque:
            //
            //      - No se esta chequeando correctamente la cantidad de parametros que trae
            //        cada servidor, el responsable de esta capa envio menos del tamaño de
            //        la capa. Notar que si enviara mas no fallaria pero no podemos garantizar
            //        que los datos sean correctos.
            //
            //      - El modelo pidio de mas, digamos que le pinto cualquiera.
            //
            panic!("Took more parameters than the ones available for this layer");
        }

        let (head, tail) = sv_slice.split_at_mut(n);
        self.slices[server_id] = tail;
        Some(head)
    }
}

/// The reversed servers' parameter iterator.
pub struct BackIter<'pm> {
    ordering: &'pm [usize],
    slices: Vec<&'pm mut [f32]>,
    curr: usize,
}

impl<'pm> BackIter<'pm> {
    /// Takes the next batch of `n` parameters following the manager's reversed ordering.
    ///
    /// # Arguments
    /// * `n` - The amount of parameters to take.
    ///
    /// # Returns
    /// A slice of parameters or `None` if the iteration has ended.
    pub fn take(&mut self, n: usize) -> Option<&'pm mut [f32]> {
        if self.curr == self.ordering.len() {
            return None;
        }

        let server_id = *self.ordering.get(self.ordering.len() - self.curr - 1)?;
        self.curr += 1;

        // SAFETY: `ParamManager`'s constructor validates that every
        //         value in the ordering is lower than it's length.
        let sv_slice = mem::take(&mut self.slices[server_id]);

        if n > sv_slice.len() {
            // TODO: Ver como podemos garantizar la invariante, que no se pidan mas parametros
            //       de los que de verdad se tiene.
            //
            //      Si se entra aca es porque:
            //
            //      - No se esta chequeando correctamente la cantidad de parametros que trae
            //        cada servidor, el responsable de esta capa envio menos del tamaño de
            //        la capa. Notar que si enviara mas no fallaria pero no podemos garantizar
            //        que los datos sean correctos.
            //
            //      - El modelo pidio de mas, digamos que le pinto cualquiera.
            //
            panic!("Took more parameters than the ones available for this layer");
        }

        let (head, tail) = sv_slice.split_at_mut(sv_slice.len() - n);
        self.slices[server_id] = head;
        Some(tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creating_an_invalid_manager_fails() {
        let ordering = vec![0, 2];
        assert!(ParamManager::new(ordering).is_none());
    }

    #[test]
    fn trying_to_create_an_iterator_without_filling_the_entire_params_vec_fails() {
        let ordering = vec![0, 1, 1, 0];
        let mut pm = ParamManager::new(ordering).unwrap();

        let mut sv0 = [1.0, 2.0];
        let mut sv1 = [3.0, 4.0, 5.0];

        assert!(pm.front().is_none());
        assert!(pm.back().is_none());
        pm.add(0, &mut sv0).unwrap();

        assert!(pm.front().is_none());
        assert!(pm.back().is_none());
        pm.add(1, &mut sv1).unwrap();

        assert!(pm.front().is_some());
        assert!(pm.back().is_some());
    }

    #[test]
    fn add_clear_work() {
        let ordering = vec![0, 1, 1, 0];
        let mut pm = ParamManager::new(ordering).unwrap();

        let mut sv0 = [1.0, 2.0];
        let mut sv1 = [3.0, 4.0, 5.0];

        assert_eq!(pm.params, [None, None]);
        pm.add(0, &mut sv0).unwrap();
        assert_ne!(pm.params, [None, None]);
        pm.add(1, &mut sv1).unwrap();
        assert_ne!(pm.params, [None, None]);

        pm.clear();
        assert_eq!(pm.params, [None, None]);
    }

    #[test]
    fn front_iterator() {
        let ordering = vec![0, 1, 1, 0];
        let mut pm = ParamManager::new(ordering).unwrap();

        let mut sv0 = [1.0, 2.0];
        let mut sv1 = [3.0, 4.0, 5.0];

        pm.add(0, &mut sv0).unwrap();
        pm.add(1, &mut sv1).unwrap();

        let mut front = pm.front().unwrap();

        assert_eq!(front.take(1).unwrap(), [1.0]);
        assert_eq!(front.take(2).unwrap(), [3.0, 4.0]);
        assert_eq!(front.take(1).unwrap(), [5.0]);
        assert_eq!(front.take(1).unwrap(), [2.0]);
    }

    #[test]
    fn back_iterator() {
        let ordering = vec![1, 0, 1, 2];
        let mut pm = ParamManager::new(ordering).unwrap();

        let mut sv0 = [1.0, 2.0, 3.0, 4.0];
        let mut sv1 = [5.0, 6.0, 7.0];
        let mut sv2 = [8.0];

        pm.add(0, &mut sv0).unwrap();
        pm.add(1, &mut sv1).unwrap();
        pm.add(2, &mut sv2).unwrap();

        let mut back = pm.back().unwrap();

        assert_eq!(back.take(1).unwrap(), [8.0]);
        assert_eq!(back.take(2).unwrap(), [6.0, 7.0]);
        assert_eq!(back.take(4).unwrap(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(back.take(1).unwrap(), [5.0]);
    }

    #[test]
    fn front_then_back() {
        let ordering = vec![0, 1, 2, 3];
        let mut pm = ParamManager::new(ordering).unwrap();

        let mut sv0 = [1.0];
        let mut sv1 = [2.0];
        let mut sv2 = [3.0];
        let mut sv3 = [4.0];

        pm.add(0, &mut sv0).unwrap();
        pm.add(1, &mut sv1).unwrap();
        pm.add(2, &mut sv2).unwrap();
        pm.add(3, &mut sv3).unwrap();

        let mut front = pm.front().unwrap();

        assert_eq!(front.take(1).unwrap(), [1.0]);
        assert_eq!(front.take(1).unwrap(), [2.0]);
        assert_eq!(front.take(1).unwrap(), [3.0]);
        assert_eq!(front.take(1).unwrap(), [4.0]);

        let mut back = pm.back().unwrap();

        assert_eq!(back.take(1).unwrap(), [4.0]);
        assert_eq!(back.take(1).unwrap(), [3.0]);
        assert_eq!(back.take(1).unwrap(), [2.0]);
        assert_eq!(back.take(1).unwrap(), [1.0]);
    }
}

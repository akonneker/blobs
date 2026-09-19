use super::paged_slots::{PagedSlots, SlotIterMut};
use super::reference::{CellKey, CellState};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::ops::{Index, Range};

/// Authoritative cells in sparse key pages with canonical active-key traversal.
/// Empty pages are released; memory does not grow with historical dead keys.
///
/// Physical layout is derived: canonical encodings, hashes, and replay always
/// traverse `active_keys`, so they do not depend on slot capacity or history.
#[derive(Debug, Clone, Default)]
pub struct CellStore {
    pub(super) slots: PagedSlots<Box<CellState>>,
    active_keys: BTreeSet<CellKey>,
}

impl PartialEq for CellStore {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for CellStore {}

impl CellStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.active_keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.active_keys.is_empty()
    }

    pub fn get(&self, key: &CellKey) -> Option<&CellState> {
        self.slots.get(key.0).map(Box::as_ref)
    }

    pub fn get_mut(&mut self, key: &CellKey) -> Option<&mut CellState> {
        self.slots.get_mut(key.0).map(Box::as_mut)
    }

    pub fn insert(&mut self, key: CellKey, cell: CellState) -> Option<CellState> {
        let previous = self.slots.insert(key.0, Box::new(cell)).map(|cell| *cell);
        if previous.is_none() {
            self.active_keys.insert(key);
        }
        previous
    }

    pub fn remove(&mut self, key: &CellKey) -> Option<CellState> {
        let removed = self.slots.remove(key.0).map(|cell| *cell);
        if removed.is_some() {
            self.active_keys.remove(key);
        }
        removed
    }

    pub fn keys(&self) -> std::collections::btree_set::Iter<'_, CellKey> {
        self.active_keys.iter()
    }

    pub fn iter(&self) -> CellIter<'_> {
        CellIter {
            keys: self.active_keys.iter(),
            slots: &self.slots,
        }
    }

    pub fn iter_mut(&mut self) -> CellIterMut<'_> {
        CellIterMut {
            slots: self.slots.iter_mut(),
        }
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &CellState> + '_ {
        self.iter().map(|(_, cell)| cell)
    }

    pub fn values_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut CellState> + '_ {
        self.slots.iter_mut().map(|(_, cell)| cell.as_mut())
    }

    pub fn range(&self, range: Range<CellKey>) -> CellRange<'_> {
        CellRange {
            keys: self.active_keys.range(range),
            slots: &self.slots,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn par_iter(
        &self,
    ) -> impl rayon::iter::ParallelIterator<Item = (CellKey, &CellState)> {
        self.slots
            .par_iter()
            .map(|(key, cell)| (CellKey(key), cell.as_ref()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn par_iter_mut(
        &mut self,
    ) -> impl rayon::iter::ParallelIterator<Item = (CellKey, &mut CellState)> {
        self.slots
            .par_iter_mut()
            .map(|(key, cell)| (CellKey(key), cell.as_mut()))
    }
}

impl FromIterator<(CellKey, CellState)> for CellStore {
    fn from_iter<T: IntoIterator<Item = (CellKey, CellState)>>(iter: T) -> Self {
        let mut cells = Self::new();
        for (key, cell) in iter {
            cells.insert(key, cell);
        }
        cells
    }
}

impl Index<&CellKey> for CellStore {
    type Output = CellState;

    fn index(&self, key: &CellKey) -> &Self::Output {
        self.get(key)
            .expect("cell key is absent from indexed storage")
    }
}

pub struct CellIter<'a> {
    keys: std::collections::btree_set::Iter<'a, CellKey>,
    slots: &'a PagedSlots<Box<CellState>>,
}

impl<'a> Iterator for CellIter<'a> {
    type Item = (&'a CellKey, &'a CellState);

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.keys.next()?;
        Some((
            key,
            self.slots
                .get(key.0)
                .map(Box::as_ref)
                .expect("active cell key is missing its indexed slot"),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.keys.size_hint()
    }
}

impl DoubleEndedIterator for CellIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let key = self.keys.next_back()?;
        Some((
            key,
            self.slots
                .get(key.0)
                .map(Box::as_ref)
                .expect("active cell key is missing its indexed slot"),
        ))
    }
}

impl ExactSizeIterator for CellIter<'_> {}

pub struct CellRange<'a> {
    keys: std::collections::btree_set::Range<'a, CellKey>,
    slots: &'a PagedSlots<Box<CellState>>,
}

impl<'a> Iterator for CellRange<'a> {
    type Item = (&'a CellKey, &'a CellState);

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.keys.next()?;
        Some((
            key,
            self.slots
                .get(key.0)
                .map(Box::as_ref)
                .expect("active cell key is missing its indexed slot"),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.keys.size_hint()
    }
}

impl Clone for CellRange<'_> {
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            slots: self.slots,
        }
    }
}

impl DoubleEndedIterator for CellRange<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let key = self.keys.next_back()?;
        Some((
            key,
            self.slots
                .get(key.0)
                .map(Box::as_ref)
                .expect("active cell key is missing its indexed slot"),
        ))
    }
}

impl<'a> IntoIterator for &'a CellStore {
    type Item = (&'a CellKey, &'a CellState);
    type IntoIter = CellIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct CellIterMut<'a> {
    slots: SlotIterMut<'a, Box<CellState>>,
}

impl<'a> Iterator for CellIterMut<'a> {
    type Item = (CellKey, &'a mut CellState);

    fn next(&mut self) -> Option<Self::Item> {
        self.slots
            .next()
            .map(|(key, cell)| (CellKey(key), cell.as_mut()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.slots.size_hint().1)
    }
}

impl<'a> IntoIterator for &'a mut CellStore {
    type Item = (CellKey, &'a mut CellState);
    type IntoIter = CellIterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

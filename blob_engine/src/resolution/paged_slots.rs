//! Sparse integer-keyed slots. Empty pages and directories are released.
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::BTreeMap;

const PAGE_SIZE: usize = 256;
const DIRECTORY_SIZE: usize = 256;
const DIRECTORY_SPAN: u64 = (PAGE_SIZE * DIRECTORY_SIZE) as u64;
#[derive(Debug, Clone)]
struct Page<T> {
    values: Box<[Option<T>; PAGE_SIZE]>,
    occupied: usize,
}
#[derive(Debug, Clone)]
struct Directory<T> {
    pages: Box<[Option<Page<T>>; DIRECTORY_SIZE]>,
    occupied: usize,
}

#[derive(Debug, Clone)]
pub(super) struct PagedSlots<T> {
    directories: BTreeMap<u64, Directory<T>>,
}
impl<T> Default for PagedSlots<T> {
    fn default() -> Self {
        Self {
            directories: BTreeMap::new(),
        }
    }
}
fn indices(key: u64) -> (u64, usize, usize) {
    (
        key / DIRECTORY_SPAN,
        ((key / PAGE_SIZE as u64) % DIRECTORY_SIZE as u64) as usize,
        (key % PAGE_SIZE as u64) as usize,
    )
}
impl<T> PagedSlots<T> {
    pub fn get(&self, key: u64) -> Option<&T> {
        let (directory, page, offset) = indices(key);
        self.directories.get(&directory)?.pages[page]
            .as_ref()?
            .values[offset]
            .as_ref()
    }
    pub fn get_mut(&mut self, key: u64) -> Option<&mut T> {
        let (directory, page, offset) = indices(key);
        self.directories.get_mut(&directory)?.pages[page]
            .as_mut()?
            .values[offset]
            .as_mut()
    }
    pub fn insert(&mut self, key: u64, value: T) -> Option<T> {
        let (directory, page, offset) = indices(key);
        let directory = self
            .directories
            .entry(directory)
            .or_insert_with(|| Directory {
                pages: Box::new(std::array::from_fn(|_| None)),
                occupied: 0,
            });
        let page = directory.pages[page].get_or_insert_with(|| {
            directory.occupied += 1;
            Page {
                values: Box::new(std::array::from_fn(|_| None)),
                occupied: 0,
            }
        });
        let previous = page.values[offset].replace(value);
        if previous.is_none() {
            page.occupied += 1;
        }
        previous
    }
    pub fn remove(&mut self, key: u64) -> Option<T> {
        let (index, page_index, offset) = indices(key);
        let directory = self.directories.get_mut(&index)?;
        let page = directory.pages[page_index].as_mut()?;
        let removed = page.values[offset].take()?;
        page.occupied -= 1;
        if page.occupied == 0 {
            directory.pages[page_index] = None;
            directory.occupied -= 1;
            if directory.occupied == 0 {
                self.directories.remove(&index);
            }
        }
        Some(removed)
    }
    pub fn clear(&mut self) {
        self.directories.clear();
    }
    #[cfg(test)]
    pub fn page_count(&self) -> usize {
        self.directories
            .values()
            .map(|directory| directory.occupied)
            .sum()
    }
    pub fn iter_mut(&mut self) -> SlotIterMut<'_, T> {
        SlotIterMut {
            directories: self.directories.iter_mut(),
            front: None,
            back: None,
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn par_iter(&self) -> impl ParallelIterator<Item = (u64, &T)>
    where
        T: Sync,
    {
        self.directories.par_iter().flat_map(|(directory, values)| {
            values
                .pages
                .par_iter()
                .enumerate()
                .filter_map(move |(page, values)| {
                    values.as_ref().map(|values| {
                        (
                            directory * DIRECTORY_SPAN + (page * PAGE_SIZE) as u64,
                            values,
                        )
                    })
                })
                .flat_map_iter(|(base, values)| {
                    values
                        .values
                        .iter()
                        .enumerate()
                        .filter_map(move |(offset, value)| {
                            value.as_ref().map(|v| (base + offset as u64, v))
                        })
                })
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn par_iter_mut(&mut self) -> impl ParallelIterator<Item = (u64, &mut T)>
    where
        T: Send,
    {
        self.directories
            .par_iter_mut()
            .flat_map(|(directory, values)| {
                values
                    .pages
                    .par_iter_mut()
                    .enumerate()
                    .filter_map(move |(page, values)| {
                        values.as_mut().map(|values| {
                            (
                                directory * DIRECTORY_SPAN + (page * PAGE_SIZE) as u64,
                                values,
                            )
                        })
                    })
                    .flat_map_iter(|(base, values)| {
                        values
                            .values
                            .iter_mut()
                            .enumerate()
                            .filter_map(move |(offset, value)| {
                                value.as_mut().map(|v| (base + offset as u64, v))
                            })
                    })
            })
    }
}

struct PageIterMut<'a, T> {
    base: u64,
    values: std::iter::Enumerate<std::slice::IterMut<'a, Option<T>>>,
}
impl<'a, T> PageIterMut<'a, T> {
    fn new(base: u64, values: &'a mut Page<T>) -> Self {
        Self {
            base,
            values: values.values.iter_mut().enumerate(),
        }
    }
}
impl<'a, T> Iterator for PageIterMut<'a, T> {
    type Item = (u64, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        self.values
            .find_map(|(offset, value)| value.as_mut().map(|v| (self.base + offset as u64, v)))
    }
}
impl<T> DoubleEndedIterator for PageIterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.values
            .by_ref()
            .rev()
            .find_map(|(offset, value)| value.as_mut().map(|v| (self.base + offset as u64, v)))
    }
}
struct DirectoryIterMut<'a, T> {
    base: u64,
    pages: std::iter::Enumerate<std::slice::IterMut<'a, Option<Page<T>>>>,
    front: Option<PageIterMut<'a, T>>,
    back: Option<PageIterMut<'a, T>>,
}
impl<'a, T> DirectoryIterMut<'a, T> {
    fn new((directory, values): (&u64, &'a mut Directory<T>)) -> Self {
        Self {
            base: directory * DIRECTORY_SPAN,
            pages: values.pages.iter_mut().enumerate(),
            front: None,
            back: None,
        }
    }
}
impl<'a, T> Iterator for DirectoryIterMut<'a, T> {
    type Item = (u64, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(value) = self.front.as_mut().and_then(Iterator::next) {
                return Some(value);
            }
            if let Some((page, values)) = self
                .pages
                .find_map(|(page, values)| values.as_mut().map(|values| (page, values)))
            {
                self.front = Some(PageIterMut::new(
                    self.base + (page * PAGE_SIZE) as u64,
                    values,
                ));
            } else {
                return self.back.as_mut().and_then(Iterator::next);
            }
        }
    }
}
impl<T> DoubleEndedIterator for DirectoryIterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(value) = self.back.as_mut().and_then(DoubleEndedIterator::next_back) {
                return Some(value);
            }
            if let Some((page, values)) = self
                .pages
                .by_ref()
                .rev()
                .find_map(|(page, values)| values.as_mut().map(|values| (page, values)))
            {
                self.back = Some(PageIterMut::new(
                    self.base + (page * PAGE_SIZE) as u64,
                    values,
                ));
            } else {
                return self.front.as_mut().and_then(DoubleEndedIterator::next_back);
            }
        }
    }
}

pub(super) struct SlotIterMut<'a, T> {
    directories: std::collections::btree_map::IterMut<'a, u64, Directory<T>>,
    front: Option<DirectoryIterMut<'a, T>>,
    back: Option<DirectoryIterMut<'a, T>>,
}
impl<'a, T> Iterator for SlotIterMut<'a, T> {
    type Item = (u64, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(value) = self.front.as_mut().and_then(Iterator::next) {
                return Some(value);
            }
            if let Some(directory) = self.directories.next() {
                self.front = Some(DirectoryIterMut::new(directory));
            } else {
                return self.back.as_mut().and_then(Iterator::next);
            }
        }
    }
}
impl<T> DoubleEndedIterator for SlotIterMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(value) = self.back.as_mut().and_then(DoubleEndedIterator::next_back) {
                return Some(value);
            }
            if let Some(directory) = self.directories.next_back() {
                self.back = Some(DirectoryIterMut::new(directory));
            } else {
                return self.front.as_mut().and_then(DoubleEndedIterator::next_back);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_keys_and_mixed_direction_mutation_match_an_ordered_map() {
        let mut slots = PagedSlots::default();
        let mut oracle = BTreeMap::new();
        for key in [0, 255, 256, 257, 65536, 1 << 60, u64::MAX] {
            slots.insert(key, key);
            oracle.insert(key, key);
        }
        let keys: Vec<_> = oracle.keys().copied().collect();
        let mut iterator = slots.iter_mut();
        for (front, expected) in [true, false, false, true, false, true, true]
            .into_iter()
            .zip([
                keys[0], keys[6], keys[5], keys[1], keys[4], keys[2], keys[3],
            ])
        {
            let (key, value) = if front {
                iterator.next()
            } else {
                iterator.next_back()
            }
            .unwrap();
            assert_eq!(key, expected);
            *value = value.wrapping_add(1);
            *oracle.get_mut(&key).unwrap() = *value;
        }
        assert!(iterator.next().is_none());
        assert!(iterator.next_back().is_none());
        for (key, expected) in oracle {
            assert_eq!(slots.remove(key), Some(expected));
        }
        assert_eq!(slots.page_count(), 0);
        assert!(slots.directories.is_empty());
    }
    #[test]
    fn birth_death_churn_releases_historical_pages() {
        let mut slots = PagedSlots::default();
        for birth in 0..20_000_u64 {
            let key = (1 << 60) + birth * 512;
            assert_eq!(slots.insert(key, birth), None);
            assert_eq!(slots.insert(key, birth + 1), Some(birth));
            if birth >= 8 {
                assert!(slots.remove((1 << 60) + (birth - 8) * 512).is_some());
            }
            assert!(slots.page_count() <= 8);
        }
        assert_eq!(slots.iter_mut().count(), 8);
        slots.clear();
        assert_eq!(slots.page_count(), 0);
        assert!(slots.directories.is_empty());
    }
}

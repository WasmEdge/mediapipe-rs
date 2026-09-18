/// Implement the list API for a newtype over `Vec<$item>`.
macro_rules! impl_result_list {
    ( $list:ident, $item:ty ) => {
        impl $list {
            pub fn len(&self) -> usize {
                self.0.len()
            }

            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            pub fn get(&self, index: usize) -> Option<&$item> {
                self.0.get(index)
            }

            pub fn first(&self) -> Option<&$item> {
                self.0.first()
            }

            pub fn last(&self) -> Option<&$item> {
                self.0.last()
            }

            pub fn iter(&self) -> std::slice::Iter<'_, $item> {
                self.0.iter()
            }

            pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, $item> {
                self.0.iter_mut()
            }

            pub fn as_slice(&self) -> &[$item] {
                self.0.as_slice()
            }

            pub fn into_inner(self) -> Vec<$item> {
                self.0
            }
        }

        impl AsRef<[$item]> for $list {
            fn as_ref(&self) -> &[$item] {
                self.0.as_slice()
            }
        }

        impl std::ops::Index<usize> for $list {
            type Output = $item;

            fn index(&self, index: usize) -> &$item {
                &self.0[index]
            }
        }

        impl std::ops::IndexMut<usize> for $list {
            fn index_mut(&mut self, index: usize) -> &mut $item {
                &mut self.0[index]
            }
        }

        impl From<Vec<$item>> for $list {
            fn from(items: Vec<$item>) -> Self {
                Self(items)
            }
        }

        impl From<$list> for Vec<$item> {
            fn from(list: $list) -> Self {
                list.0
            }
        }

        impl IntoIterator for $list {
            type Item = $item;
            type IntoIter = std::vec::IntoIter<$item>;

            fn into_iter(self) -> Self::IntoIter {
                self.0.into_iter()
            }
        }

        impl<'a> IntoIterator for &'a $list {
            type Item = &'a $item;
            type IntoIter = std::slice::Iter<'a, $item>;

            fn into_iter(self) -> Self::IntoIter {
                self.0.iter()
            }
        }

        impl<'a> IntoIterator for &'a mut $list {
            type Item = &'a mut $item;
            type IntoIter = std::slice::IterMut<'a, $item>;

            fn into_iter(self) -> Self::IntoIter {
                self.0.iter_mut()
            }
        }
    };
}
pub(crate) use impl_result_list;

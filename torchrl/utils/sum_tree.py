class SumTree:
  """Implements a Sum Tree data structure.

  A Sum Tree data structure is a binary tree
  with leaf nodes containing data and internal
  nodes containing sum of the tree rooted at
  that node. The binary tree here is represented
  by an array.
  """
  def __init__(self, capacity: int = 16):
    assert not (capacity & (capacity - 1)), \
      "Capacity should be a power of two, found {}".format(capacity)

    self.capacity = capacity
    self.tree = None
    self._next_target_index = 0

    self.clear()

  def add(self, value):
    self.update(self._next_target_index, value)
    self._next_target_index = (self._next_target_index + 1) % self.capacity

  def update(self, index: int, value):
    tree_index = self.capacity - 1 + index
    delta = value - self.tree[tree_index]

    if delta:
      iter_index = tree_index
      while iter_index >= 0:
        self.tree[iter_index] += delta

        iter_index = (iter_index - 1) // 2

  def get(self, value):
    search_value = value
    tree_index = 0

    while True:
      left_child = 2 * tree_index + 1
      right_child = 2 * tree_index + 2

      if left_child >= len(self):
        break

      if search_value <= self.tree[left_child]:
        tree_index = left_child
      else:
        search_value -= self.tree[left_child]
        tree_index = right_child

    return tree_index - (self.capacity - 1)

  def clear(self):
    self.tree = [0.0] * (2 * self.capacity - 1)
    self._next_target_index = 0

  def __len__(self):
    return len(self.tree)

  @property
  def next_free_index(self):
    return self._next_target_index

  @property
  def max_value(self):
    return max(self.tree[-self.capacity:])

  @property
  def sum_value(self):
    return self.tree[0]

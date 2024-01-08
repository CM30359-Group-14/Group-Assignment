import operator

from typing import Callable


class SegmentTree:
    """
    Class representing a SegmentTree. 
    
    This code has been more or less copied from OpenAI baselines:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    The point of re-writing it is to make sure I have an understanding of how SegmentTrees work.
    """

    def __init__(self, capacity: int, operation: Callable, neutral_value: float):
        """
        Instantiates a SegmentTree.

        :param capacity: the maximum capacity of the tree.
        :param operation: the operation performed by calling `reduce`.
        :param neutral_value: ...
        """
        self.capacity = capacity
        self.tree = [neutral_value for _ in range(2 * capacity)]
        self.operation = operation

    def _reduce_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        """
        Returns the result of `self.operation` in segment.
        """
        if start == node_start and end == node_end:
            return self.tree[node]
        
        mid = (node_start + node_end) // 2

        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )        

    def reduce(self, start: int, end: int = 0) -> float:
        """
        Returns the result of applying `self.operation` to a contiguous
        subsequence of the array.

        :param start: the beginning of the subsequence.
        :param end: the end of the subsequence.
        """
        if end <= 0:
            end += self.capacity
        end -= 1
        
        return self._reduce_helper(start, end, 1, 0, self.capacity - 1)
    
    def __setitem__(self, idx: int, val: float):
        """
        Sets a value in tree.
        """
        # Index of the leaf
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """
        Gets the real value in the corresponding leaf node of tree.
        """
        return self.tree[self.capacity + idx]
    

class SumSegmentTree(SegmentTree):
    """
    Class representing a segment tree of sums.
    """
    
    def __init__(self, capacity: int):
        """
        Instantaites a SumSegmentTree.

        :param capacity: the maximum capacity of the tree.
        """
        super().__init__(capacity, operator.add, 0.0)

    def sum(self, start: int = 0, end: int = 0) -> float:
        """
        Returns arr[start] + ... + arr[end].
        """
        return self.reduce(start, end)
    
    def retrieve(self, upperbound: float) -> int:
        """
        Finds the highest index `i` about upper bound in the tree.
        """
        idx = 1

        # While non-leaf node.
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1

            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right

        return idx - self.capacity
    

class MinSegmentTree(SegmentTree):
    """
    Class representing a segment tree of minimums.
    """

    def __init__(self, capacity: int):
        """
        Instantiates a MinSegmentTree.

        :param capacity: the maximum capacity of the tree.
        """
        super().__init__(capacity, min, float("inf"))

    def min(self, start: int = 0, end: int = 0) -> float:
        """
        Returns min(arr[start], ..., arr[end])
        """
        return self.reduce(start, end)
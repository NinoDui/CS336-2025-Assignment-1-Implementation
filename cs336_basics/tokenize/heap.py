from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class MaxHeap(Generic[T]):
    def __init__(
        self, data: Iterable[T] | None = None, key: Callable[[T], Any] = lambda x: x
    ):
        super().__init__()

        self._heap = []
        self._key = key

        if data is not None:
            match data:
                case list():
                    self._heap = data
                case dict():
                    self._heap = list(data.items())
                case _:
                    self._heap = list(data)
            self.heapify()

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self) == 0

    def _move_up(self, idx: int):
        parent_idx = (idx - 1) // 2
        while idx > 0 and self._key(self._heap[idx]) > self._key(
            self._heap[parent_idx]
        ):
            self._heap[idx], self._heap[parent_idx] = (
                self._heap[parent_idx],
                self._heap[idx],
            )
            idx = parent_idx
            parent_idx = (idx - 1) // 2

    def _move_down(self, idx: int):
        n = len(self._heap)
        left_idx = (idx << 1) + 1
        right_idx = (idx << 1) + 2

        largest_idx = idx
        if left_idx < n and self._key(self._heap[left_idx]) > self._key(
            self._heap[idx]
        ):
            largest_idx = left_idx
        if right_idx < n and self._key(self._heap[right_idx]) > self._key(
            self._heap[largest_idx]
        ):
            largest_idx = right_idx

        if largest_idx != idx:
            self._heap[idx], self._heap[largest_idx] = (
                self._heap[largest_idx],
                self._heap[idx],
            )
            self._move_down(largest_idx)

    def push(self, item: T):
        self._heap.append(item)
        self._move_up(len(self._heap) - 1)

    def pop(self) -> T:
        if self.is_empty():
            raise IndexError("Heap is empty")

        root = self._heap[0]
        self._heap[0] = self._heap[-1]
        self._heap.pop()
        self._move_down(0)

        return root

    def peek(self) -> T:
        if self.is_empty():
            raise IndexError("Heap is empty")
        return self._heap[0]

    def heapify(self):
        N = len(self._heap)
        start_idx = (N - 1) // 2  # last non-leaf node
        for idx in range(start_idx, -1, -1):
            self._move_down(idx)

    def __repr__(self) -> str:
        return f"MaxHeap(N={len(self._heap)}, key={self._key})"

    def __str__(self) -> str:
        return self.__repr__()

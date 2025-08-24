from cs336_basics.tokenize.heap import MaxHeap


def test_max_heap():
    data = {
        ("l", "o", "w"): 5,
        ("l", "o", "w", "e", "r"): 2,
        ("w", "i", "d", "e", "s", "t"): 3,
    }

    heap = MaxHeap(data, key=lambda x: x[-1])

    assert heap.peek() == (("l", "o", "w"), 5)
    assert heap.pop() == (("l", "o", "w"), 5)
    assert heap.peek() == (("w", "i", "d", "e", "s", "t"), 3)
    heap.push((("n", "e", "w", "e", "r"), 7))
    assert heap.peek() == (("n", "e", "w", "e", "r"), 7)

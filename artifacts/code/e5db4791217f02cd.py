
def bubble_sort(arr):
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# tests
assert bubble_sort([5, 3, 1, 4, 2]) == [1, 2, 3, 4, 5]
assert bubble_sort([])              == []
assert bubble_sort([1])             == [1]
assert bubble_sort([2, 1])          == [1, 2]
print("bubble_sort: all tests passed")

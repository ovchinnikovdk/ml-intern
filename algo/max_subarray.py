

def findMaxSubArray(A):
    psums = [0, A[0]]
    for i in range(1, len(A)):
        psums.append(A[i] + psums[-1])
    max_sum = psums[1]
    max_subarr = [0, 1]
    for i in range(len(A)):
        j = i + 1
        while j <= len(A):
            subsum = psums[j] - psums[i]
            if subsum > max_sum:
                max_sum = subsum
                max_subarr = [i, j]
            j += 1
    return A[max_subarr[0]:max_subarr[1]]


with open('arrays.txt') as f:
    arrays = [[int(num.strip()) for num in line.split(',')] for line in f.read().splitlines()]
    for array in arrays:
        print(findMaxSubArray(array))

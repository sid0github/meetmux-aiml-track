name = "Siddharth"
age = 21
marks = [85, 92, 78, 95, 88]

# accsessing elements
for i in marks:
    print(f"validating data points: {i}")

# function
def analytics_marks(numbers):
    min_num  = min(numbers)
    max_num = max(numbers)
    avg_num = sum(numbers) / len(numbers)
    print(sorted(numbers))
    median_num = sorted(numbers)[len(numbers) // 2]  # Calculate median
    return min_num, max_num, avg_num, median_num

results = analytics_marks(marks)  #results is a tuple containing min, max and avg marks
print(f"Minimum marks: {results[0]}")   
print(f"Maximum marks: {results[1]}")
print(f"Average marks: {results[2]}")
print(f"Median marks: {results[3]}")





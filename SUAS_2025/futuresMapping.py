from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time


def print_numbers():
    for i in range(10):
        print(f"Number: {i}")
        time.sleep(1)


def print_letters():
    for letter in 'abcdefghij':
        print(f"Letter: {letter}")
        time.sleep(1)


# Using ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(print_numbers), executor.submit(print_letters)]
    for future in futures:
        future.result()

print("All tasks completed with ThreadPoolExecutor")

# Using ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(print_numbers), executor.submit(print_letters)]
    for future in futures:
        future.result()

print("All tasks completed with ProcessPoolExecutor")
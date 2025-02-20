def generate_all_binary_numbers(N):
    print(f"generate_all_binary_numbers({N}) çağrıldı")
    if N == 0:
        print(f"Base condition met: generate_all_binary_numbers({N}) returns ['']")
        return ['']
    
    child_results = generate_all_binary_numbers(N-1)
    print(f"N = {N} için child_results: {child_results}")
    results = []
    
    for prefix in ('0', '1'):
        for suffix in child_results:
            new_result = prefix + suffix
            print(f"Yeni sonuç oluşturuluyor: {prefix} + {suffix} = {new_result}")
            results.append(new_result)
    
    print(f"generate_all_binary_numbers({N}) returns {results}")
    return results

# Example usage
N = 2
binary_numbers = generate_all_binary_numbers(N)
print(f"All binary numbers of length {N}: {binary_numbers}")


print(binary_numbers)


def generate_permutations(string):
    print(f"generate_permutations('{string}') çağrıldı")
    # Base case: if the string is empty or has only one character, return it
    if len(string) <= 1:
        print(f"Base condition met: generate_permutations('{string}') returns ['{string}']")
        return [string]
    
    # Recursive case
    permutations = []
    for i, char in enumerate(string):
        # Remove the current character from the string
        remaining_chars = string[:i] + string[i+1:]
        print(f"Karakter '{char}' çıkarıldıktan sonra kalan string: '{remaining_chars}'")
        
        # Recursively generate permutations for the remaining characters
        sub_permutations = generate_permutations(remaining_chars)
        print(f"generate_permutations('{remaining_chars}') returns {sub_permutations}")
        
        # Add the current character to the beginning of each sub-permutation
        for sub_perm in sub_permutations:
            new_perm = char + sub_perm
            print(f"Yeni permütasyon oluşturuluyor: {char} + {sub_perm} = {new_perm}")
            permutations.append(new_perm)
    
    print(f"generate_permutations('{string}') returns {permutations}")
    return permutations

# Example usage
string = "abc"
permutations = generate_permutations(string)
print(f"All permutations of '{string}': {permutations}")





def factorial(n):
    print(f"factorial({n}) çağrıldı")
    # Base case: if n is 0 or 1, return 1
    if n == 0:
        print(f"Base condition met: factorial({n}) returns 1")
        return 1
    
    result = n * factorial(n - 1)
    print(f"factorial({n}) returns {result}")
    return result

# Test the function
print(f"factorial(5) sonucu: {factorial(5)}")  # Output should be 120

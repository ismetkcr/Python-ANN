def cascading_influence(influences, T):
    """
    Simulate the cascading influence in a network over time.

    Args:
    influences (list): Influence strengths for each step.
    T (int): Total number of time steps.

    Returns:
    list: A list of total influences for each time step.
    """
    total_influence = []

    # Outer loop iterates over each node/time step
    for t in range(T):
        I_t = 0  # Initialize influence for time step t

        # Inner loop, summing influences starting from t+2
        for k in range(t + 2, T):  # Start from t+2, going up to T-1
            I_t += influences[k - t - 2] / (k)  # Adjust index for influences and divide by k

        total_influence.append(I_t)  # Store the total influence for node t

    return total_influence

# Example usage:
influences = [5, 3, 2, 1, 0.5]  # Example influence strengths
T = len(influences)  # Total number of steps

total_influence = cascading_influence(influences, T)
print("Total Influence for each time step:", total_influence)

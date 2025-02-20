import numpy as np


Ca_sat = 96.35294117647058
Ca_1 = 41.80882353
Ca_2 = 74.63235294

numerator = Ca_sat - Ca_1
denumerator = Ca_sat - Ca_2

kla = np.log(numerator/denumerator) / 250


#dont using this
def calculate_kLa_2(C_oks, max_C, step=10):
    """
    Calculate kLa values for the given C_oks data array and step size.
    Assumes kLa = 0 if log argument is invalid (e.g., Ca1, Ca2, and Ca_sat are equal).

    Parameters:
    - C_oks (np.ndarray): The array of C_oks data.
    - step (int): The step size for calculating differences (default is 10).

    Returns:
    - np.ndarray: An array of kLa values.
    """
    # Ensure the input is a NumPy array
    C_oks = np.array(C_oks)

    # Determine C_a_sat (max element of C_oks)
    Ca_sat = max_C

    # Initialize kLa array
    kLa_values = []

    # Loop through the array, stepping by the given step size
    for i in range(0, len(C_oks) - step, step):
        Ca_1 = C_oks[i]
        Ca_2 = C_oks[i + step]
        # Ensure valid logarithm input
        numerator = Ca_sat - Ca_1
        denominator = Ca_sat - Ca_2

        if numerator > 0 and denominator > 0:
            kLa = np.log(numerator / denominator) / step
        else:
            kLa = 0  # Assume 0 if invalid


        kLa_values.append(kLa)
        #print(Ca_sat, Ca_1, Ca_2, Ca_2-Ca_1, kLa) for debugging


    # Convert the result to a NumPy array
    return np.array(kLa_values)
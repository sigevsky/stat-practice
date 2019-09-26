import numpy as np
import pandas as pd
from shared import *
from scipy.stats import ttest_ind, f_oneway

alpha = 0.005


def main():
    data = {
            test: [fractal_dimension(np.load(f"raw/{E_OPEN}/{i}/{test}.npy"))
              for i in range(1, STUDENTS_NUM + 1)]
                for test in TESTS
          }
    df = pd.DataFrame.from_dict(data)
    p_values = [[ttest_ind(df[i], df[j], equal_var=False)[1] for i in TESTS] for j in TESTS]
    statistics = [[ttest_ind(df[i], df[j], equal_var=False)[0] for i in TESTS] for j in TESTS]
    df.to_csv("raw/fractal_dimensions.csv")

    res_inter_state = f_oneway(df[TESTS[0]], df[TESTS[1]], df[TESTS[2]], df[TESTS[3]], df[TESTS[4]])
    res_inter_students = f_oneway(*df.as_matrix())

    print("Inter student one way ANOVA test")
    print(res_inter_students)
    print("Inter state one way ANOVA test")
    print(res_inter_state)

    print("------T-tests for inter state samples------")
    print("P-values:")
    print(np.array(p_values))
    print("Statistics:")
    print(np.array(statistics))

    print("-------Original dataframe--------")
    print(df)


def fractal_dimension(Z):
    def box_count(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(box_count(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

main()
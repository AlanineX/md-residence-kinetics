import numpy as np
import pandas as pd

# pKa values
pKa1 = 10.7  # BH+ -> B + H+
pKa2 = 7.6   # BH2^2+ -> BH+ + H+

Ka1 = 10**(-pKa1)
Ka2 = 10**(-pKa2)

# pH range with gap = 0.5
pH_values = np.arange(0.0, 14.0, 0.5)

data = []

for pH in pH_values:
    H = 10**(-pH)

    denom = H**2 + Ka2*H + Ka2*Ka1

    alpha_BH2 = H**2 / denom
    alpha_BH1 = Ka2*H / denom
    alpha_B   = Ka2*Ka1 / denom

    data.append([
        pH,
        alpha_BH2 * 100,
        alpha_BH1 * 100,
        alpha_B * 100
    ])

df = pd.DataFrame(
    data,
    columns=[
        "pH",
        "BH2^2+ (%)",
        "BH+ (%)",
        "B (%)"
    ]
)

print(df)

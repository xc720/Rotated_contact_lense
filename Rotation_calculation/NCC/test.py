import numpy as np

# Your list of y-values
y = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5])

# Compute differences between consecutive y-values
dy = np.diff(y)

# Compute the sign of the differences
signs = np.sign(dy)

# Find where the sign changes
sign_changes = np.where(np.diff(signs) != 0)[0] + 1

print("Indices where the derivative changes sign:", sign_changes)

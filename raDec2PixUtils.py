
import numpy as np
import array
        
def bound(val, lower, upper):
    val = np.array(val);
    
    while np.sum(val < lower) > 0:
        val[val<lower] = val[val<lower] + (upper - lower)

    while np.sum(val > upper) > 0:
        val[val>upper] = val[val>upper] - (upper - lower)

    return val

def make_col(x):
    if x.size == 1:
        return x;
    else:
        return np.reshape(x, (x.size,1));

def append_col(a1, col):
  if np.any(a1 == None):
      return col;
  else:
      return np.column_stack((a1, col));

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


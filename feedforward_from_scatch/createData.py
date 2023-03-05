import numpy as np
import random
import matplotlib.pyplot as plt

def createData(nr_class1, nr_class2):
    outer_radius = 1
    inner_radius = 0.5
    
    # angle, radius
    alpha = np.random.uniform(low = 0, high = 2 * np.pi, size = (nr_class1,))
    radius = np.random.uniform(low = inner_radius, high = outer_radius, size = (nr_class1,))
    
    # create class 1 (outer ring)
    class1_x = radius * np.cos(alpha)
    class1_y = radius * np.sin(alpha)
    
    # new random angles and radius for class 2
    alpha = np.random.uniform(low = 0, high = 2 * np.pi, size = (nr_class2,))
    radius = np.random.uniform(low = 0, high = inner_radius, size = (nr_class2,))
    
    # create class 2 (inner circle)
    class2_x = radius * np.cos(alpha)
    class2_y = radius * np.sin(alpha)

    # reshape
    class1 = np.column_stack((class1_x, class1_y))
    class2 = np.column_stack((class2_x, class2_y))
    
    # shape
    return class1, class2
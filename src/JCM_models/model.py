import numpy as np
import matplotlib.pyplot as plt

class Shape:
    def __init__(self, name, domain_id, priority, side_length_constraint, points,nk):
        self.name = name
        self.domain_id = domain_id
        self.priority = priority
        self.side_length_constraint = side_length_constraint
        self.points = points
        self.nk = nk

        self.permittivity = np.square(self.nk)

    def describe(self):
        return f"""Shape: {self.name}
  Domain ID: {self.domain_id}
  Priority: {self.priority}
  Side Length Constraint: {self.side_length_constraint}
  Refractive Index (nk): {self.nk}
  Permittivity (ε): {self.permittivity}
"""
    def plot(self, ax=None, **kwargs):
        x = self.points[::2]
        y = self.points[1::2]
        x = np.append(x, x[0])  # Close the polygon
        y = np.append(y, y[0])
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_title(f"Shape: {self.name}")
            ax.legend()
            
            ax.plot(x, y, label=self.name, **kwargs)
            return fig
        else:
            ax.plot(x, y, label=self.name, **kwargs)
            return ax
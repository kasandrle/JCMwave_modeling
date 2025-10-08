import math
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from .utils import corner_round
import numpy as np

class ShapeGenerator:
    """
    🔷 ShapeGenerator: A modular geometry engine for parametric shape creation.

    Supports multiple shape types including:
    - Rectangle
    - Trapezoid
    - Stacked trapezoids
    - B-splines

    Features:
    • Flexible parameter dictionary per shape
    • Optional offset for x and y positioning
    • Corner rounding via arc interpolation
    • Centering and flattening utilities
    • Matplotlib-based plotting
    • Ceremonial narration via `.describe()`

    Shape-specific parameters:
    - Rectangle: height, width
    - Trapezoid: height, width, side_angle_deg
    - Stacked trapezoids: height (list), width (list of len+1)
    - B-splines: control_points, num_points

    Example:
        sg = ShapeGenerator('rectangle', {'height': 10, 'width': 20})
        sg.plot()
        print(sg.describe())
    """

    def __init__(self, shape_type, params, offset_x=0, offset_y=0):
        self.shape_type = shape_type.lower()
        self.params = params
        self.offset_x = offset_x
        self.offset_y = offset_y

    def generate(self):
        shape_map = {
            'rectangle': self._rectangle,
            'stack_trapezoids': self._stack_trapezoids,
            'trapezoid': self._trapezoid,
            'bsplines': self._bsplines
        }
        shape_func = shape_map.get(self.shape_type)
        if not shape_func:
            raise ValueError(f"Unsupported shape type: {self.shape_type}")

        coords = shape_func()

        # Check for embedded corner radii
        corner_radii = self.params.get('corner_radii')
        if corner_radii:
            coords = self._apply_corner_radii(coords, corner_radii)

        return coords


    def flatten(self,centered= False):
        if centered:
            coords = self.centered()
        else:
            coords = self.generate()
        return [coord for point in coords for coord in point]

    def centered(self):
        """Returns coordinates with x-axis centered at zero."""
        coords = self.generate()
        x_vals = [pt[0] for pt in coords]
        x_center = (max(x_vals) + min(x_vals)) / 2
        return [(x - x_center+self.offset_x, y) for (x, y) in coords]
    
    def _apply_corner_radii(self, coords, corner_radii, n=50):
        """Applies per-corner rounding based on corner_radii dict."""
        rounded_coords = []
        for i in range(len(coords)):
            if i in corner_radii:
                r = corner_radii[i]
                x1 = coords[i - 1]
                x2 = coords[i]
                x3 = coords[(i + 1) % len(coords)]
                arc = corner_round(x1, x2, x3, r, n)
                rounded_coords.extend(arc)
            else:
                rounded_coords.append(coords[i])
        return rounded_coords

    def _stack_trapezoids(self):
        heights = self.params.get('height')
        widths = self.params.get('width')  # length must be len(heights) + 1

        if not (isinstance(heights, list) and isinstance(widths, list)):
            raise ValueError("'height' and 'width' must be lists")
        if len(widths) != len(heights) + 1:
            raise ValueError("Length of 'width' must be one more than 'height'")

        shapes_right = []
        shapes_left = []
        y_cursor = self.offset_y

        for i in range(len(heights)):
            h = heights[i]
            bottom_w = widths[i]
            top_w = widths[i + 1]

            # Coordinates from bottom left, clockwise
            coords_right = [
                (self.offset_x + bottom_w / 2, y_cursor),
                (self.offset_x + top_w / 2, y_cursor + h),
            ]
            coords_left = [
                (self.offset_x - bottom_w / 2, y_cursor),
                (self.offset_x - top_w / 2, y_cursor + h)
            ]

            shapes_right.extend(coords_right)
            shapes_left.extend(coords_left)
            y_cursor += h 
        unique_right = sorted(set(shapes_right), key=lambda pt: pt[1])  # bottom to top
        unique_left = sorted(set(shapes_left), key=lambda pt: pt[1], reverse=True)  # top to bottom

        return unique_right + unique_left

    def _bsplines(self):
        control_points = self.params.get('control_points')
        num_points = self.params.get('num_points',200)
        if control_points is None:
            raise ValueError("Besplines requires 'control_points'")
        control_points = np.array(control_points)
        x_points, y_points = control_points[:, 0], control_points[:, 1]
        # Knot vector (uniform, clamped)
        degree = 3  # Cubic B-spline for smoothness
        num_ctrl_pts = len(control_points)
        knots = np.concatenate(([0] * (degree + 1), np.arange(1, num_ctrl_pts - degree), [num_ctrl_pts - degree] * (degree + 1)))

        # Parameter values for sampling
        u = np.linspace(knots[degree], knots[-(degree + 1)], num_points)

        # B-spline basis functions for x and y
        bspline_x = BSpline(knots, x_points, degree)
        bspline_y = BSpline(knots, y_points, degree)

        # Evaluate the B-spline
        x_spline = bspline_x(u)
        y_spline = bspline_y(u)

        coords = []
        for i in range(len(x_spline)):
            coords.append((self.offset_x + x_spline[i], self.offset_y + y_spline[i]))

        return coords



    def _rectangle(self):
        h = self.params.get('height')
        w = self.params.get('width')
        if h is None or w is None:
            raise ValueError("Rectangle requires 'height' and 'width'")

        return [            
            (self.offset_x, self.offset_y),
            (self.offset_x + w, self.offset_y),
            (self.offset_x + w, self.offset_y + h),
            (self.offset_x, self.offset_y + h),     
        ]


    def _trapezoid(self):
        h = self.params.get('height')
        mid_w = self.params.get('width')
        angle_deg = 90-self.params.get('side_angle_deg')
        if h is None or mid_w is None or angle_deg is None:
            raise ValueError("Trapezoid requires 'height', 'width', and 'side_angle_deg' 'width' is at half height")

        angle_rad = math.radians(angle_deg)
        half_height = h / 2
        offset = half_height * math.tan(angle_rad)

        bottom_w = mid_w + 2 * offset
        top_w = mid_w - 2 * offset

        # Clamp top width to zero if negative
        top_w = max(0, top_w)

        # Coordinates from bottom left, clockwise
        return [
            
            (self.offset_x + bottom_w / 2, self.offset_y),
            (self.offset_x + top_w / 2, self.offset_y + h),
            (self.offset_x - top_w / 2, self.offset_y + h),
            (self.offset_x - bottom_w / 2, self.offset_y),
        ]
    

    def plot(self, ax=None, title=None):
        coords = self.generate()
        x, y = zip(*coords)
        if ax is None:
            fig  = plt.figure(figsize=(6, 6))
            plt.plot(x, y, marker='o', linestyle='-', color='teal')
            plt.fill(x, y, alpha=0.3, color='skyblue')
            plt.title(title or f"{self.shape_type.capitalize()} Shape")
            plt.axis('equal')
            plt.grid(True)
            plt.show()
            return fig
        else:
            ax.plot(x, y, marker='o', linestyle='-', color='teal')
            ax.fill(x, y, alpha=0.3, color='skyblue')
            return ax


    def describe(self):
        lines = [f"🔷 Shape: {self.shape_type.capitalize()}"]

        # Shape-specific parameter keys
        shape_params = {
            'rectangle': ['height', 'width'],
            'trapezoid': ['height', 'width', 'side_angle_deg'],
            'stack_trapezoids': ['height', 'width'],
            'bsplines': ['control_points', 'num_points']
        }

        used_keys = shape_params.get(self.shape_type, [])
        for key in used_keys:
            if key in self.params:
                lines.append(f"• {key.replace('_', ' ').capitalize()}: {self.params[key]}")

        # Stacked trapezoid narration
        if self.shape_type == 'stack_trapezoids':
            heights = self.params.get('height', [])
            widths = self.params.get('width', [])
            print(heights)
            if isinstance(heights, list) and isinstance(widths, list) and len(widths) == len(heights) + 1:
                lines.append("• Layer transitions:")
                for i in range(len(heights)):
                    lines.append(f"   - Layer {i}: height {heights[i]}, bottom width {widths[i]}, top width {widths[i+1]}")

        # Corner rounding
        corner_radii = self.params.get('corner_radii')
        if corner_radii:
            if self.shape_type == 'rectangle':
                raw_coords = self._rectangle()
            elif self.shape_type == 'trapezoid':
                raw_coords = self._trapezoid()
            elif self.shape_type == 'stack_trapezoids':
                raw_coords = self._stack_trapezoids()
            else:
                raw_coords = []

            lines.append("• Rounded corners:")
            for i, r in corner_radii.items():
                if i < len(raw_coords):
                    x, y = raw_coords[i]
                    lines.append(f"   - Corner {i} at ({x:.2f}, {y:.2f}) → radius {r}")
                else:
                    lines.append(f"   - Corner {i} → radius {r} (index out of bounds)")

        # Offset
        if self.offset_x != 0 or self.offset_y != 0:
            lines.append(f"• Offset: x = {self.offset_x}, y = {self.offset_y}")

        # Centering hint
        lines.append("• Centered option available via `.centered()`")

        return "\n".join(lines)
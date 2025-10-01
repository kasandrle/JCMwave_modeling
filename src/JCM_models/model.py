import numpy as np
import matplotlib.pyplot as plt
from .utils import eVnm_converter

class Shape:
    """
    🔷 Shape: A domain-aware polygonal object with optical properties.

    Represents a geometric shape defined by:
    • A name and domain ID
    • A priority for simulation layering
    • A side length constraint (for meshing or physical limits)
    • A list of 2D points (flattened [x0, y0, x1, y1, ...])
    • A complex refractive index (nk)
    • Boundary conditions per edge (default: ['Transparent','Periodic','Transparent','Periodic'])

    Automatically computes:
    • Permittivity (ε) as nk²

    Methods:
    • describe(): returns a summary of the shape's identity and optical properties
    • plot(ax=None, **kwargs): visualizes the shape as a closed polygon using Matplotlib

    Example:
        shape = Shape(
            name='Slab',
            domain_id=1,
            priority=0,
            side_length_constraint=1.0,
            points=[-1, -1, 1, -1, 1, 1, -1, 1],
            nk=2.0 + 0.1j
        )
        shape.plot()
        print(shape.describe())
    """

    def __init__(self, name, domain_id, priority, side_length_constraint, points,nk,boundary = ['Transparent','Periodic','Transparent','Periodic']):
        self.name = name
        self.domain_id = domain_id
        self.priority = priority
        self.side_length_constraint = side_length_constraint
        self.points = points
        self.nk = nk
        self.boundary = boundary

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
        
class Source:
    """
    🔆 Source: A physically grounded illumination object for optical simulation.

    Represents an incident wave defined by:
    • Wavelength (`lam`) in nm, eV, or meters
    • Polarization vector: [1, 0] → S-polarized, [0, 1] → P-polarized
    • Angle of incidence (θ) in degrees
    • Azimuthal angle (phi) in degrees
    • Direction of incidence: 'FromAbove' or 'FromBelow'
    • Type of source: default is 'PlaneWave'

    Automatically converts wavelength to meters and validates input formats.

    Methods:
    • polarization_label(): returns 'S', 'P', or 'Mixed or custom'
    • describe(): narrates all physical parameters in ceremonial format

    Example:
        src = Source(
            lam=532,
            polarization=[1, 0],
            angle_of_incidence=45,
            phi=0,
            incidence='FromAbove',
            unit='nm'
        )
        print(src.describe())
    """

    def __init__(self, lam, polarization, angle_of_incidence, phi, incidence='FromAbove', unit='nm',type = 'PlaneWave'):
        allowed = {'FromAbove', 'FromBelow'}
        if incidence not in allowed:
            raise ValueError(f"incidence must be one of {allowed}, got '{incidence}'")

        if not (isinstance(polarization, list) and len(polarization) == 2):
            raise ValueError("polarization must be a list of two numbers")

        if unit == 'nm':
            self.lam = lam*1e-9
        elif unit == 'eV':
            self.lam = eVnm_converter(lam)*1e-9
        elif unit == 'm':
            self.lam = lam
        else:
            raise ValueError(f"unit must be 'nm', 'eV', or 'm', got '{unit}'")

        self.polarization = polarization
        self.angle_of_incidence = angle_of_incidence
        self.phi = phi
        self.incidence = incidence
        self.type = type

    def polarization_label(self):
        if self.polarization == [1, 0]:
            return 'S'
        elif self.polarization == [0, 1]:
            return 'P'
        else:
            return 'Mixed or custom'

    def describe(self):
        lines = [f"🔆 Source description:"]
        lines.append(f"• Wavelength: {self.lam}")
        lines.append(f"• Polarization: {self.polarization} → {self.polarization_label()}-polarized")
        lines.append(f"• Angle of incidence: {self.angle_of_incidence}°")
        lines.append(f"• Azimuthal angle (phi): {self.phi}°")
        lines.append(f"• Incidence direction: {self.incidence}")
        lines.append(f"• Type: {self.type}")
        return "\n".join(lines)
    

class Cartesian:
    """
    📐 Cartesian grid definition for field export.

    You must specify *either*:
    • NGridPointsX / NGridPointsY (discrete grid definition), OR
    • Spacing (uniform spacing in meters)

    Not both at the same time.
    """

    def __init__(self, spacing=None, n_grid_points_x=None, n_grid_points_y=None):
        if spacing is not None and (n_grid_points_x or n_grid_points_y):
            raise ValueError("Specify either spacing OR NGridPoints, not both.")

        if spacing is None and (n_grid_points_x is None or n_grid_points_y is None):
            raise ValueError("If spacing is not given, both NGridPointsX and NGridPointsY must be provided.")

        self.spacing = spacing
        self.n_grid_points_x = n_grid_points_x
        self.n_grid_points_y = n_grid_points_y

    def describe(self):
        if self.spacing is not None:
            return f"📐 Cartesian grid with spacing = {self.spacing} m"
        else:
            return f"📐 Cartesian grid with NGridPointsX={self.n_grid_points_x}, NGridPointsY={self.n_grid_points_y}"

    def to_dict(self):
        if self.spacing is not None:
            return {"Cartesian": {"Spacing": self.spacing}}
        else:
            return {
                "Cartesian": {
                    "NGridPointsX": self.n_grid_points_x,
                    "NGridPointsY": self.n_grid_points_y,
                }
            }
        
    def to_jcm(self, indent=2):
        pad = " " * indent
        lines = [f"{pad}Cartesian {{"]

        if self.spacing is not None:
            lines.append(f"{pad}  Spacing = {self.spacing}")
        else:
            lines.append(f"{pad}  NGridPointsX = {self.n_grid_points_x}")
            lines.append(f"{pad}  NGridPointsY = {self.n_grid_points_y}")

        lines.append(f"{pad}}}")
        return "\n".join(lines)


class PostProcess:
    """
    🌀 PostProcess: Ritual container for simulation field analysis.

    Two distinct modes are supported:

    • ExportFields:
        - field_bag_path (str)
        - output_file_name (str)
        - output_quantity (str)
        - domain_ids (list[int], optional)
        - cartesian (dict, optional) e.g. {"Spacing": 0.1e-9}

    • FourierTransform:
        - field_bag_path (str)
        - output_file_name (str)
        - normal_direction (str: 'X','Y','Z')
        - rotation (str, optional)

    Methods:
    • describe(): narrates the chosen post-process in ceremonial format
    """

    def __init__(self, mode,field_bag_path,output_file_name, **kwargs):
        allowed_modes = {"ExportFields", "FourierTransform"}
        if mode not in allowed_modes:
            raise ValueError(f"mode must be one of {allowed_modes}, got '{mode}'")

        self.mode = mode
        self.field_bag_path = field_bag_path
        self.output_file_name = output_file_name

        if mode == "ExportFields":
            required = ["output_quantity"]
            for r in required:
                if r not in kwargs:
                    raise ValueError(f"Missing required parameter '{r}' for ExportFields")

            self.output_quantity = kwargs["output_quantity"]
            self.domain_ids = kwargs.get("domain_ids")
            self.cartesian = kwargs.get("cartesian")

        elif mode == "FourierTransform":
            self.normal_direction = kwargs.get("normal_direction")
            self.rotation = kwargs.get("rotation")

    def describe(self):
        lines = [f"🌀 PostProcess description:"]
        lines.append(f"• Mode: {self.mode}")
        lines.append(f"• FieldBagPath: {self.field_bag_path}")
        lines.append(f"• OutputFileName: {self.output_file_name}")

        if self.mode == "ExportFields":
            lines.append(f"• OutputQuantity: {self.output_quantity}")
            if self.domain_ids:
                lines.append(f"• DomainIds: {self.domain_ids}")
            if self.cartesian:
                lines.append(self.cartesian.describe())

        elif self.mode == "FourierTransform":
            lines.append(f"• NormalDirection: {self.normal_direction}")
            if self.rotation:
                lines.append(f"• Rotation: {self.rotation}")

        return "\n".join(lines)

    def to_jcm(self, indent=0):
        pad = " " * indent
        lines = [f"{pad}PostProcess {{", f"{pad}  {self.mode} {{"]

        # Shared fields
        lines.append(f'{pad}    FieldBagPath = "{self.field_bag_path}"')
        lines.append(f'{pad}    OutputFileName = "{self.output_file_name}"')

        if self.mode == "ExportFields":
            lines.append(f'{pad}    OutputQuantity = "{self.output_quantity}"')
            if self.domain_ids:
                ids = ", ".join(map(str, self.domain_ids))
                lines.append(f"{pad}    DomainIds = [{ids}]")
            if self.cartesian:
                lines.append(self.cartesian.to_jcm(indent + 4))

        elif self.mode == "FourierTransform":
            if self.normal_direction:
                lines.append(f"{pad}    NormalDirection = {self.normal_direction}")
            if self.rotation:
                lines.append(f"{pad}    Rotation = {self.rotation}")

        lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return "\n".join(lines)
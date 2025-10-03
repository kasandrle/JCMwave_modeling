import os

def write_file(folder_name, filename, content):
    """
    Create a folder (if it doesn't exist) and write a file with given content.
    """
    os.makedirs(folder_name, exist_ok=True)
    filepath = os.path.join(folder_name, filename)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"{filename} written to {filepath}")


def write_project_files(folder_name):
    # --- project.jcmpt ---
    project_content = """Project = {
  InfoLevel = 1
  Electromagnetics {
      TimeHarmonic {
               Scattering {
        FieldComponents = Electric
        FiniteElementDegree = %(fem_deg)i


       }

                  }
  }
}

<?
for postprocess in keys['postprocess']:
    keys['postprocess'] = postprocess.to_jcm()
    ?>

    %(postprocess)s
    """
    write_file(folder_name, "project.jcmpt", project_content)

    # --- source.jcmt ---
    source_content = """
<?
for i, source in enumerate(keys['source']):
  keys['lam'] = source.lam
  keys['pol'] = source.polarization
  keys['angle_of_incidence'] = source.angle_of_incidence
  keys['phi'] = source.phi
  keys['incidence'] = source.incidence
  ?>
  SourceBag {

  Source {

      ElectricFieldStrength {
        PlaneWave {
          Lambda0 = %(lam)e
          SP = %(pol)e 
          ThetaPhi = [%(angle_of_incidence)e, %(phi)e]
          3DTo2D = yes
          Incidence = %(incidence)s
        }
      }
                      <?
  ?>
    }
  }
"""
    write_file(folder_name, "source.jcmt", source_content)

    # --- layout.jcm ---
    layout_content = """
Layout {
Name = "periodic_line"
UnitOfLength = %(uol1)e


<?
for i, shape in enumerate(keys['shape']):
    keys['id'] = shape.domain_id
    keys['point'] = shape.points
    keys['prior'] = shape.priority
    keys['layer'] = shape.name
    keys['slc'] = shape.side_length_constraint
    if shape.name == "ComputationalDomain":
        ?>
        Polygon {
            Name = "%(layer)s"
            DomainId = %(id)e
            Priority = %(prior)e
            Points = %(point)e
            SideLengthConstraint = %(slc)e
        <?
        for ii,boundary in enumerate(shape.boundary):
            keys['BoundaryClass'] = boundary
            keys['Number'] = ii+1
            ?>
            BoundarySegment {
            Number = %(Number)e
            BoundaryClass = %(BoundaryClass)s
            }
        <?
        ?>
        }
        <?
        ?>

    <?
    else:
        ?>
        Polygon {
                Name = "%(layer)s"
                DomainId = %(id)e
                Priority = %(prior)e
                Points = %(point)e
                SideLengthConstraint = %(slc)e
            }
        <?
?>
}
"""
    write_file(folder_name, "layout.jcmt", layout_content)

    # --- materials.jcm ---
    materials_content = """
<?

for shape in keys['shape']:
    if shape.name == 'ComputationalDomain':
      keys['permittivity_'] = shape.permittivity
    else:
      keys['permittivity_'] = shape.permittivity[keys['energy_index']]
    keys['RelPermeability_'] = 1 #keys['RelPermeability'][n]
    keys['name_'] = shape.name
    keys['id_'] = shape.domain_id
    

    ?>
    Material {
      Name = "%(name_)s"
      Id = %(id_)i
      RelPermittivity = %(permittivity_)e
      RelPermeability = %(RelPermeability_)e
}"""
    write_file(folder_name, "materials.jcmt", materials_content)


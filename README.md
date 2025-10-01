# JCMwave_modeling

## Define shape array with multiple Shape instances
shape = [
    Shape(
        'ComputationalDomain',
        domain_id=1,
        priority=-1,
        side_length_constraint=slc,
        points=computional_domain,
        nk=1,
        boundary=['Transparent', 'Periodic', 'Transparent', 'Periodic']
    ),
    Shape(
        'substrate',
        domain_id=2,
        priority=1,
        side_length_constraint=slc,
        points=substrate,
        nk=nk_sub
    ),
    Shape(
        'substrate_oxide',
        domain_id=3,
        priority=2,
        side_length_constraint=slc,
        points=substrate_oxide,
        nk=nk_sub_oxid
    ),
    Shape(
        'resist',
        domain_id=5,
        priority=2,
        side_length_constraint=slc,
        points=trapzoid_stack,
        nk=nk_resist
    ),
]


## Assign shape and source arrays to keys dictionary
keys['shape'] = shape
keys['source'] = [s_eV]
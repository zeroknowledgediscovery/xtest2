"""Static periodic-table reference data used only for deterministic chemistry
summary features derived from `formula`.

Values are standard textbook constants (Pauling electronegativity, empirical
atomic radius in picometers, IUPAC group number, period). Lanthanides and
actinides are assigned group 3 by the usual convention. These numbers are
fixed reference constants, not fitted from the JARVIS catalog, and do not
depend on any split of the data.

Columns: (atomic_number, electronegativity, atomic_radius_pm, group, period)
"""

ELEMENT_DATA = {
    "H":  (1, 2.20, 25, 1, 1),
    "He": (2, 0.00, 120, 18, 1),
    "Li": (3, 0.98, 145, 1, 2),
    "Be": (4, 1.57, 105, 2, 2),
    "B":  (5, 2.04, 85, 13, 2),
    "C":  (6, 2.55, 70, 14, 2),
    "N":  (7, 3.04, 65, 15, 2),
    "O":  (8, 3.44, 60, 16, 2),
    "F":  (9, 3.98, 50, 17, 2),
    "Ne": (10, 0.00, 160, 18, 2),
    "Na": (11, 0.93, 180, 1, 3),
    "Mg": (12, 1.31, 150, 2, 3),
    "Al": (13, 1.61, 125, 13, 3),
    "Si": (14, 1.90, 110, 14, 3),
    "P":  (15, 2.19, 100, 15, 3),
    "S":  (16, 2.58, 100, 16, 3),
    "Cl": (17, 3.16, 100, 17, 3),
    "Ar": (18, 0.00, 71, 18, 3),
    "K":  (19, 0.82, 220, 1, 4),
    "Ca": (20, 1.00, 180, 2, 4),
    "Sc": (21, 1.36, 160, 3, 4),
    "Ti": (22, 1.54, 140, 4, 4),
    "V":  (23, 1.63, 135, 5, 4),
    "Cr": (24, 1.66, 140, 6, 4),
    "Mn": (25, 1.55, 140, 7, 4),
    "Fe": (26, 1.83, 140, 8, 4),
    "Co": (27, 1.88, 135, 9, 4),
    "Ni": (28, 1.91, 135, 10, 4),
    "Cu": (29, 1.90, 135, 11, 4),
    "Zn": (30, 1.65, 135, 12, 4),
    "Ga": (31, 1.81, 130, 13, 4),
    "Ge": (32, 2.01, 125, 14, 4),
    "As": (33, 2.18, 115, 15, 4),
    "Se": (34, 2.55, 115, 16, 4),
    "Br": (35, 2.96, 115, 17, 4),
    "Kr": (36, 3.00, 88, 18, 4),
    "Rb": (37, 0.82, 235, 1, 5),
    "Sr": (38, 0.95, 200, 2, 5),
    "Y":  (39, 1.22, 180, 3, 5),
    "Zr": (40, 1.33, 155, 4, 5),
    "Nb": (41, 1.60, 145, 5, 5),
    "Mo": (42, 2.16, 145, 6, 5),
    "Tc": (43, 1.90, 135, 7, 5),
    "Ru": (44, 2.20, 130, 8, 5),
    "Rh": (45, 2.28, 135, 9, 5),
    "Pd": (46, 2.20, 140, 10, 5),
    "Ag": (47, 1.93, 160, 11, 5),
    "Cd": (48, 1.69, 155, 12, 5),
    "In": (49, 1.78, 155, 13, 5),
    "Sn": (50, 1.96, 145, 14, 5),
    "Sb": (51, 2.05, 145, 15, 5),
    "Te": (52, 2.10, 140, 16, 5),
    "I":  (53, 2.66, 140, 17, 5),
    "Xe": (54, 2.60, 108, 18, 5),
    "Cs": (55, 0.79, 260, 1, 6),
    "Ba": (56, 0.89, 215, 2, 6),
    "La": (57, 1.10, 195, 3, 6),
    "Ce": (58, 1.12, 185, 3, 6),
    "Pr": (59, 1.13, 185, 3, 6),
    "Nd": (60, 1.14, 185, 3, 6),
    "Pm": (61, 1.13, 185, 3, 6),
    "Sm": (62, 1.17, 185, 3, 6),
    "Eu": (63, 1.20, 185, 3, 6),
    "Gd": (64, 1.20, 180, 3, 6),
    "Tb": (65, 1.10, 175, 3, 6),
    "Dy": (66, 1.22, 175, 3, 6),
    "Ho": (67, 1.23, 175, 3, 6),
    "Er": (68, 1.24, 175, 3, 6),
    "Tm": (69, 1.25, 175, 3, 6),
    "Yb": (70, 1.10, 175, 3, 6),
    "Lu": (71, 1.27, 175, 3, 6),
    "Hf": (72, 1.30, 155, 4, 6),
    "Ta": (73, 1.50, 145, 5, 6),
    "W":  (74, 2.36, 135, 6, 6),
    "Re": (75, 1.90, 135, 7, 6),
    "Os": (76, 2.20, 130, 8, 6),
    "Ir": (77, 2.20, 135, 9, 6),
    "Pt": (78, 2.28, 135, 10, 6),
    "Au": (79, 2.54, 135, 11, 6),
    "Hg": (80, 2.00, 150, 12, 6),
    "Tl": (81, 1.62, 190, 13, 6),
    "Pb": (82, 2.33, 180, 14, 6),
    "Bi": (83, 2.02, 160, 15, 6),
    "Po": (84, 2.00, 190, 16, 6),
    "At": (85, 2.20, 150, 17, 6),
    "Rn": (86, 2.20, 150, 18, 6),
    "Fr": (87, 0.70, 260, 1, 7),
    "Ra": (88, 0.90, 215, 2, 7),
    "Ac": (89, 1.10, 195, 3, 7),
    "Th": (90, 1.30, 180, 3, 7),
    "Pa": (91, 1.50, 180, 3, 7),
    "U":  (92, 1.38, 175, 3, 7),
    "Np": (93, 1.36, 175, 3, 7),
    "Pu": (94, 1.28, 175, 3, 7),
}

ATOMIC_NUMBER = {k: v[0] for k, v in ELEMENT_DATA.items()}
ELECTRONEGATIVITY = {k: v[1] for k, v in ELEMENT_DATA.items()}
ATOMIC_RADIUS = {k: v[2] for k, v in ELEMENT_DATA.items()}
GROUP = {k: v[3] for k, v in ELEMENT_DATA.items()}
PERIOD = {k: v[4] for k, v in ELEMENT_DATA.items()}

# Space groups (International Tables numbering) that are centrosymmetric,
# i.e. whose point group / Laue class contains inversion. This is a fixed
# crystallographic fact table, not derived from the JARVIS catalog.
CENTROSYMMETRIC_SPG_RANGES = [
    (2, 2),
    (10, 15),
    (47, 74),
    (83, 88),
    (123, 142),
    (147, 148),
    (162, 167),
    (175, 176),
    (191, 194),
    (200, 206),
    (221, 230),
]


def is_centrosymmetric(spg_number: int) -> int:
    for lo, hi in CENTROSYMMETRIC_SPG_RANGES:
        if lo <= spg_number <= hi:
            return 1
    return 0

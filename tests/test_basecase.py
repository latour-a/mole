"""
Teste les fonctions du module basecase.
"""

import numpy as np
import basecase as bc

def test_admissible():
    "Teste la fonction `admissible` du module basecase."
    # Cas basiques :
    grid = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    got = bc.admissible(grid, threshold=2)
    assert (got == True)
    grid = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]])
    got = bc.admissible(grid, threshold=2)
    assert (got == False)
    # Une seule dimension :
    grid = np.array([0, 0, 1, 0, 0, 1, 0, 0])
    got = bc.admissible(grid, threshold=3)
    assert (got == True)
    grid = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    got = bc.admissible(grid, threshold=3)
    assert (got == False)
    # Seuil > taille :
    grid = np.array([[0, 0], [0, 0]])
    got = bc.admissible(grid, threshold=4)
    assert (got == True)
    grid = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    got = bc.admissible(grid, threshold=4)
    assert (got == False)
    # Plus de 2 dimensions :
    grid = np.ones(24, dtype=np.int).reshape((4, 3, 2))
    got = bc.admissible(grid, threshold=1)
    assert (got == True)
    grid = np.ones(24, dtype=np.int).reshape((4, 3, 2))
    grid[1, :, 1] = 0
    got = bc.admissible(grid, threshold=3)
    assert (got == False)

def test_score():
    "Teste la fonction `score` du module basecase."
    # Cas basiques :
    grid = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    got = bc.score(grid, threshold=2)
    assert (got == 4)
    grid = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]])
    got = bc.score(grid, threshold=2)
    assert (got == np.inf)
    # Une seule dimension :
    grid = np.array([0, 0, 1, 0, 0, 1, 0, 0])
    got = bc.score(grid, threshold=3)
    assert (got == 2)
    grid = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    got = bc.score(grid, threshold=3)
    assert (got == np.inf)
    # Seuil > taille :
    grid = np.array([[0, 0], [0, 0]])
    got = bc.score(grid, threshold=5)
    assert (got == 0)
    # Plus de 2 dimensions :
    grid = np.ones(24, dtype=np.int).reshape((4, 3, 2))
    got = bc.score(grid, threshold=1)
    assert (got == 24)
    grid = np.ones(24, dtype=np.int).reshape((4, 3, 2))
    grid[1, :, 1] = 0
    got = bc.score(grid, threshold=3)
    assert (got == np.inf)

def test_generate():
    "Teste la fonction `generate` du module basecase."
    # Sans pièges imposés :
    expected = np.zeros((2, 3), dtype=np.int)
    got = bc.generate((2, 3), npoints=0)
    assert (np.all(got == expected) == True)
    # Avec autant de pièges imposés que de cases :
    expected = np.ones((2, 3), dtype=np.int)
    got = bc.generate((2, 3), npoints=6)
    assert (np.all(got == expected) == True)
    # Nombre de pièges :
    got = bc.generate((10, 5), npoints=20)
    assert (got.sum() == 20)

# coding: utf8
"""
Teste les fonctions du module basecase.
"""

import pytest
import numpy as np
from mole import basecase as bc

def test_admissible():
    "Teste la fonction `admissible` du module basecase."
    # Cas basiques :
    grid = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    got = bc.admissible(grid, threshold=2)
    assert (got is True)
    grid = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]])
    got = bc.admissible(grid, threshold=2)
    assert (got is False)
    # Une seule dimension :
    grid = np.array([0, 0, 1, 0, 0, 1, 0, 0])
    got = bc.admissible(grid, threshold=3)
    assert (got is True)
    grid = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    got = bc.admissible(grid, threshold=3)
    assert (got is False)
    # Seuil > taille :
    grid = np.array([[0, 0], [0, 0]])
    got = bc.admissible(grid, threshold=4)
    assert (got is True)
    grid = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    got = bc.admissible(grid, threshold=4)
    assert (got is False)
    # Plus de 2 dimensions :
    grid = np.ones(24, dtype=np.int).reshape((4, 3, 2))
    got = bc.admissible(grid, threshold=1)
    assert (got is True)
    grid = np.ones(24, dtype=np.int).reshape((4, 3, 2))
    grid[1, :, 1] = 0
    got = bc.admissible(grid, threshold=3)
    assert (got is False)

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
    assert np.all(got == expected)
    # Avec autant de pièges imposés que de cases :
    expected = np.ones((2, 3), dtype=np.int)
    got = bc.generate((2, 3), npoints=6)
    assert np.all(got == expected)
    # Nombre de pièges :
    got = bc.generate((10, 5), npoints=20)
    assert (got.sum() == 20)
    # Levée d'exception :
    with pytest.raises(ValueError):
        bc.generate((10, 0, 3), npoints=0)

def test_solve():
    "Teste la fonction `solve` du module basecase."
    # 1 dimension :
    grid = np.array([1, 0, 0, 0])
    got = bc.solve(grid, 2, '1dim')
    expected = np.array([1, 0, 1, 0])
    assert np.all(got == expected)
    # 2 dimensions, carré, solution unique :
    grid = np.array([[1, 0], [0, 0]])
    got = bc.solve(grid, 2, '2dims-unicity')
    expected = np.array([[1, 0], [0, 1]])
    assert np.all(got == expected)
    # 2 dimensions, nombre de pièges connu (mais solution non unique) :
    threshold = 3
    grid = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
    got = bc.solve(grid, threshold, '2dims-square')
    assert bc.admissible(got, threshold)
    assert (bc.score(got, threshold) == 8)
    # 2 dimensions, non-carré :
    threshold = 3
    grid = np.array([[0, 0, 0], [0, 1, 0]])
    got = bc.solve(grid, threshold, '2dims-rect')
    assert bc.admissible(got, threshold)
    assert (bc.score(got, threshold) == 2)
    # 2 dimensions, pièges déjà placés :
    threshold = 3
    grid = np.array([[0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 1, 0, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]])
    got = bc.solve(grid, threshold, '2dims-preset')
    assert bc.admissible(got, threshold)
    assert bc.score(got, threshold) == 8
    expected = np.array([[0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [1, 1, 0, 1, 1],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0]])
    assert np.all(got == expected)
    # 2 dimensions, déjà résolu :
    grid = np.array([[0, 0], [0, 0]])
    got = bc.solve(grid, 3, '2dims-solved')
    assert np.all(got == grid)
    # 2 dimensions, déjà résolu :
    grid = np.array([[1, 0], [0, 1]])
    got = bc.solve(grid, 2, '2dims-solved')
    assert np.all(got == grid)

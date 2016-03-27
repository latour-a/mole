# coding: utf8
"""
Teste les fonctions du module basecase.
"""

import tempfile
import numpy as np
import makedata as mk

class MockPb():
    """
    Fausse classe implémentant les fonctions attendues des modules `pb` utilisés
    dans makedata.
    """

    def __init__(self, shape, npoints, threshold):
        "Initialise une nouvelle instance de la classe `MockPb`."
        self.shape = shape
        self.npoints = npoints
        self.threshold = threshold
        self._grid = np.random.randint(low=0, high=2, size=shape)
        self._solution = np.random.randint(low=0, high=2, size=shape)
        while np.all(self._grid == self._solution):
            self._solution = np.random.randint(low=0, high=2, size=shape)

    def generate(self, shape, npoints):
        "Renvoie une fausse instance."
        return self._grid.copy()

    def admissible(self, grid, threshold):
        "Indique si `grid` est une solution."
        return np.all(grid == self._solution)

    def solve(self, grid, threshold, name):
        "Renvoie une fausse solution."
        return self._solution.copy()

def test_makeone():
    "Teste la création d'une donnée via le module makedata."
    pb = MockPb((5, 5), 0, 3)
    params = mk.InstanceParams(pb.shape, pb.npoints, pb.threshold)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Création d'une donnée :
        output1 = mk.makeone(pb, params, tmpdir)
        grid, solution = mk.load(output1)
        got = pb.admissible(grid, params.threshold)
        assert (got == False)
        got = pb.admissible(solution, params.threshold)
        assert (got == True)
        # Création d'une autre donnée : même avec les mêmes paramètres, elle
        # doit être sauvegardé dans un fichier distinct.
        output2 = mk.makeone(pb, params, tmpdir)
        assert (output1 != output2)

def test_fmt():
    "Teste l'homogénéisation des données lors de l'enregistrement."
    pb = MockPb((2, 5, 3), 1, 2)
    params = mk.InstanceParams(pb.shape, pb.npoints, pb.threshold)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Création d'une donnée :
        output = mk.makeone(pb, params, tmpdir)
        grid, solution = mk.load(output)
        assert (grid.shape == (5, 3, 2)) # Les axes ont été triés par taille.
        assert (solution.shape == (5, 3, 2))

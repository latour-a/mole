# coding: utf8
"""
Teste les fonctions du module basecase.
"""

import time
import shutil
import tempfile
import numpy as np
from mole import makedata as mk

class TemporaryDirectory(object):
    """
    Context manager pour gérer un répertoire temporaire (implémenté dans le
    module `tempfile` en Python 3).
    """

    def __init__(self):
        self.name = tempfile.mkdtemp()

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        shutil.rmtree(self.name)

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

def genparams(shape, threshold):
    "Fonction créant un générateur de paramètres aléatoires."
    while True:
        npoints = np.random.binomial(np.product(shape), 0.5)
        yield mk.InstanceParams(shape, npoints, threshold)

def test_makeone():
    "Teste la création d'une donnée via le module makedata."
    pb = MockPb((5, 5), 0, 3)
    params = mk.InstanceParams(pb.shape, pb.npoints, pb.threshold)
    with TemporaryDirectory() as tmpdir:
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
    with TemporaryDirectory() as tmpdir:
        # Création d'une donnée :
        output = mk.makeone(pb, params, tmpdir)
        grid, solution = mk.load(output)
        assert (grid.shape == (5, 3, 2)) # Les axes ont été triés par taille.
        assert (solution.shape == (5, 3, 2))

def test_makeseveral():
    "Teste la création de plusieurs données via le module makedata."
    pb = MockPb((5, 5), 0, 3)
    params = mk.InstanceParams(pb.shape, pb.npoints, pb.threshold)
    nsamples = 2
    maxtime = 0.5
    with TemporaryDirectory() as tmpdir:
        # Création de deux données :
        res = mk.makeseveral(pb, params, tmpdir, nsamples=nsamples)
        assert (len(res) == nsamples)
        # Utilisation d'une limite en temps :
        start = time.time()
        res = mk.makeseveral(pb, params, tmpdir, maxtime=maxtime)
        assert (len(res) >= 1)
        assert ((time.time() - start) >= maxtime)
        # Utilisation simultanée d'une limite en temps et en nombre de données :
        start = time.time()
        res = mk.makeseveral(pb, params, tmpdir, nsamples=nsamples, maxtime=maxtime)
        got = (len(res) == nsamples) or ((time.time() - start) > maxtime)
        assert (got == True)
        # Utilisation de paramètres aléatoires :
        res = mk.makeseveral(pb, genparams((5, 5), 3), tmpdir, nsamples=nsamples)
        assert (len(res) == nsamples)



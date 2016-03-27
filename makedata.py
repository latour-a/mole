# coding: utf8
"""
Ensemble de fonctions pour créer des ensembles de données d'apprentissage pour
la résolution du problème "Le jardinier et les taupes".
"""

import os
import itertools
import numpy as np
from uuid import uuid4
from collections import namedtuple

InstanceParams = namedtuple('InstanceParams', ('shape', 'npoints', 'threshold'))

def _argsort(it, **kwargs):
    """
    Renvoie une version triée de l'itérable `it`, ainsi que les indices
    correspondants au tri.

    Paramètres :
    ------------
    - it : itérable
    - kwargs
        Mots-clés et valeurs utilisables avec la fonction built-in `sorted`.

    Résultats :
    -----------
    - indexes : itérable d'indices
        Indices des éléments de `it`, dans l'ordre dans lequel les éléments
        apparaissent dans `sortedit`.
    - sortedit : itérable
        Version triée de `it`.

    Exemples :
    ----------
    >>> it = [2, 1, 3]
    >>> indexes, sortedit = _argsort(it)
    >>> indexes
    (1, 0, 2)
    >>> sortedit
    (1, 2, 3)
    >>> [it[x] for x in indexes]
    [1, 2, 3]
    >>> indexes, sortedit = _argsort(it, reverse=True)
    >>> indexes
    (2, 0, 1)
    >>> sortedit
    (3, 2, 1)
    """
    indexes, sortedit = zip(*sorted(enumerate(it), key=lambda x: x[1],
                                    **kwargs))
    return indexes, sortedit

def _save(name, params, grid, solution, where):
    """
    Sauvegarde sur disque l'instance `grid` du problème "Le jardinier et les
    taupes", ainsi que sa solution `solution`.

    Paramètres :
    ------------
    - name : chaîne de caractères
        Nom de l'instance `grid`.
    - params : InstanceParams
        Paramètres utilisés pour générer `grid`.
    - grid : tableau numpy
        Instance du problème "Le jardinier et les taupes".
    - solution : tableau numpy
        Solution du problème "Le jardinier et les taupes" correspondant à
        l'instance `grid`.
    - where : chaîne de caractères
        Dossier dans lequel stocker `grid` et `solution`. L'organisation
        interne de `where` est gérée par la fonction.
    """
    # Homogénéisation de la taille des problèmes (i.e. : une grille de taille
    # 5x3 est équivalente à sa transposée de taille 3x5) :
    indexes, shape = _argsort(params.shape, reverse=True)
    grid = grid.transpose(indexes)
    solution = solution.transpose(indexes)
    # Utilisation d'une règle de tri sommaire :
    shapekey = "x".join([str(i) for i in shape])
    thresholdkey = "threshold" + str(params.threshold)
    outputdir = os.path.join(where, thresholdkey, shapekey)
    try:
        os.makedirs(outputdir) # Lève une exception si le dossier existe déjà.
    except OSError:
        pass
    outputname = os.path.join(outputdir, name) + '.npz'
    # Sauvegarde dans un format numpy :
    np.savez(outputname, grid=grid, solution=solution)
    return outputname

def load(fname):
    """
    Renvoie le contenu du fichier `fname` :
    - une instance du problème "Le jardinier et les taupes" ;
    - une de ses solutions optimales.
    """
    data = np.load(fname)
    return data["grid"], data["solution"]

def makeone(pb, params, where):
    """
    Crée une donnée pour l'apprentissage du problème "Le jardinier et les
    taupes" :
        1. Génère une instance du problème dans la version du module `pb`,
           pour les paramètres `params`.
        2. Résout cette instance.
        3. Sauvegarde la grille et sa solution.
    Renvoie le nom du fichier où est enregistré la donnée.

    Paramètres :
    ------------
    - pb
        Objet permettant de générer et de résoudre une instance du problème.
    - params : InstanceParams
        Paramètres de l'instance à générer.
    - where : chaîne de caractères
        Dossier dédié au stockage des données pour le problème `pb`.
    """
    # Création d'un nom unique :
    name = str(uuid4())
    # Génération de l'instance :
    grid = pb.generate(params.shape, params.npoints)
    # Résolution de l'instance :
    if pb.admissible(grid, params.threshold):
        solution = grid
    else:
        solution = pb.solve(grid, params.threshold, name)
    # Sauvegarde :
    if pb.admissible(solution, params.threshold):
        return _save(name, params, grid, solution, where)
    else:
        raise ValueError("failed to solve %s." % name)

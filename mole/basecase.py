# coding: utf8
"""
Ensemble de fonctions pour manipuler une instance du problème "Le jardinier et
les taupes", dans le cas où l'objectif est :
  - qu'aucune taupe ne puisse pénétrer dans le jardin ;
  - que le nombre de pièges soit minimal.
"""

import os
import itertools
import numpy as np
import pulp

def _dimcheck(grid, threshold):
    """
    Indique si la grille dont `grid` est la complémentaire est admissible
    dans sa dernière dimension.

    Paramètres :
    ------------
    - grid : tableau numpy
        Tableau contenant des 0 (espace occupé) et des 1 (espace libre).
    - threshold : entier positif
        Nombre d'espaces libres adjacents à partir duquel la grille est
        considérée comme non admissible.

    Exemples :
    ----------
    # On crée une grille avec deux espaces libres adjacents :
    >>> grid = np.array([0, 1, 1, 0, 0])
    >>> _dimcheck(grid, 2)
    False
    >>> _dimcheck(grid, 3)
    True
    """
    dsize = grid.shape[-1]
    if threshold > dsize:
        return True
    elif threshold < 0:
        raise ValueError("threshold must be positive.")
    check = 0
    for start in range(threshold):
        check += grid[..., start:(dsize - threshold + 1 + start)]
    if np.any(check >= threshold):
        return False
    else:
        return True

def admissible(grid, threshold):
    """
    Indique si la grille `grid` est admissible. Une grille est admissible si
    elle ne contient jamais plus de `threshold` - 1 espaces libres adjacents.

    Paramètres :
    ------------
    - grid : tableau numpy
        Tableau contenant des 0 (espace libre) et des 1 (espace occupé).
    - threshold : entier positif
        Nombre d'espaces libres adjacents à partir duquel la grille est
        considérée comme non admissible.

    Exemples :
    ----------
    >>> grid = np.array([0, 1, 1, 0, 0, 1]).reshape((2, 3))
    >>> admissible(grid, 2)
    False
    >>> admissible(grid, 3)
    True
    """
    # La méthode de calcul est bourrine.
    comp = np.where(grid, 0, 1) # On travaille sur le complémentaire de grid.
    res = True
    for _ in range(comp.ndim):
        res = (res and _dimcheck(comp, threshold))
        if res is False:
            break
        # Permutation circulaire des axes :
        comp = comp.transpose(comp.ndim - 1, *range(comp.ndim - 1))
    return res

def score(grid, threshold):
    """
    Calcule le score associé à la grille `grid`. Plus le score est faible,
    meilleur il est ; si la grille n'est pas admissible, renvoie un score
    égal à l'infini.

    Paramètres :
    ------------
    - grid : tableau numpy
        Tableau contenant des 0 (espace libre) et des 1 (espace occupé).
    - threshold : entier positif
        Nombre d'espaces libres adjacents à partir duquel la grille est
        considérée comme non admissible.
    """
    if admissible(grid, threshold):
        return grid.sum()
    else:
        return np.inf

def generate(shape, npoints):
    """
    Génère une grille ayant la forme `shape` et contenant `npoints` pièges.

    Paramètres :
    ------------
    - shape : entier positif, tuple d'entiers positifs
        Dimensions de la grille.
    - npoints : entier positif
        Nombre de pièges imposés à placer aléatoirement dans la grille.
    """
    size = np.product(shape)
    if size <= 0:
        raise ValueError("the shape %s should contain positive values only."\
                         % str(shape))
    points = np.random.choice(np.arange(size), npoints, replace=False)
    grid = np.zeros(size, dtype=np.int)
    grid[points] = 1
    return grid.reshape(shape)

def _key(indexes, fromstr=False):
    """
    Convertit `indexes` :
    - d'un tuple d'indices vers une chaîne de caractères (`fromstr == False`) ;
    - d'une chaîne de caractères vers un tuple d'indices (`fromstr == True`) .
    """
    sep = '_'
    if fromstr:
        return tuple([int(idx) for idx in indexes.split(sep)])
    else:
        return sep.join((str(idx) for idx in indexes))

def _slice(index, i, nneighbors, lim):
    """
    A partir de l'indice `index`, renvoie les `nneighbors` indices suivants
    dans la dimension `i` (y compris `index`, qui est le premier élément de
    la liste).

    Si certains des indices calculés sont supérieurs à `lim` dans la dimension
    `i`, renvoie une liste vide.

    Paramètres :
    ------------
    - index : tuple d'indices
    - i : entier
        Dimension à considérer.
    - nneighbors : entier positif
        Nombre d'indices à considérer dans la dimension `i` à partir de
        l'indice `index`.
    - lim : entier positif
        Indice maximal dans la dimension `i`.

    Exemples :
    ----------
    >>> index = (1, 2, 3)
    >>> _slice(index, i=0, nneighbors=4, lim=4)
    [(1, 2, 3), (2, 2, 3), (3, 2, 3), (4, 2, 3)]
    >>> _slice(index, i=0, nneighbors=4, lim=3)
    []
    >>> _slice(index, i=1, nneighbors=2, lim=10)
    [(1, 2, 3), (1, 3, 3)]
    """
    rng = range(nneighbors)
    if index[i] + rng[-1] > lim:
        return []
    return [index[:i] + (index[i] + j,) + index[(i+1):] for j in rng]

def solve(grid, threshold, name, compdir=None):
    """
    Résout le problème "Le jardinier et les taupes" pour la grille `grid`,
    avec des taupes de taille `threshold`.

    Paramètres :
    ------------
    - grid : tableau numpy
        Tableau contenant des 0 (espace libre) et des 1 (piège déjà posé).
    - threshold : entier positif
        Nombre d'espaces libres adjacents à partir duquel la grille est
        considérée comme non admissible (= taille des taupes).
    - name : chaîne de caractères
        Nom de l'instance à résoudre.
    - compdir : chaîne de caractères, None par défaut
        Dossier dans lequel effectuer les calculs (il s'agit du répertoire
        courant si `compdir` vaut `None`).

    Remarque : le nom de l'instance est utilisée comme nom pour le fichier
    d'instruction du solveur. Si la fonction doit être exécutée plusieurs
    fois en parallèle, il est nécessaire que `name` soit unique.
    """
    # Initialisation du problème :
    prob = pulp.LpProblem(name, pulp.LpMinimize)
    # Déclaration des variables :
    varnames = []
    for index in itertools.product(*(range(size) for size in grid.shape)):
        if grid[index] == 0:
            varnames.append(_key(index))
    varprefix = "Cells"
    cells = pulp.LpVariable.dicts(varprefix, varnames, 0, 1, 'Integer')
    # Déclaration de la fonction objectif :
    prob += pulp.lpSum([cells[idx] for idx in varnames]), "Non empty points"
    # Déclaration des variables et des contraintes :
    # TODO : sans faire compliqué, il y a des leviers d'amélioration :
    # - dans la boucle, il n'est pas nécessaire d'itérer jusqu'à `size` ;
    # - on peut éliminer à l'avance certaines variables (à moins que PuLP
    #   ne le fasse déjà).
    it = itertools.product(*(range(size) for size in grid.shape))
    it = filter(lambda x: grid[x] == 0, it)
    for index in it:
        for i in range(len(index)): # Itération sur toutes les dimensions.
            neighbors = _slice(index, i, threshold, grid.shape[i] - 1)
            # Si on est proche de l'extrémité de la grille (`neighbors` est
            # vide) ou s'il y a un piège proche dans toutes les directions,
            # inutile d'aller plus loin :
            if neighbors and all([_key(j) in varnames for j in neighbors]):
                prob += pulp.lpSum([cells[_key(j)] for j in neighbors]) >= 1,\
                        "Cell_%s_dim_%d" % (_key(index), i)
    # Résolution du problème:
    fname = "%s.lp" % name
    if compdir is not None:
        fname = os.path.join(compdir, fname)
    prob.writeLP(fname)
    try:
        prob.solve()
    finally:
        os.remove(fname)
    # Vérification de l'optimalité de la solution :
    status = pulp.constants.LpStatus[prob.status].lower()
    if status != 'optimal':
        raise ValueError("optimization %s did not converge." % name)
    # Mise en forme du résultat :
    res = grid.copy()
    for cell in prob.variables():
        # PuLP renomme les variables de manière assez malcommode :
        name = cell.name.replace(varprefix + '_', '')
        if name in varnames:
            res[_key(name, fromstr=True)] = cell.varValue
    return res

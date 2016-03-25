"""
Ensemble de fonctions pour manipuler une instance du problème "Le jardinier et
les taupes", dans le cas où l'objectif est :
  - qu'aucune taupe ne puisse pénétrer dans le jardin ;
  - que le nombre de pièges soit minimal.
"""

import numpy as np

def _dimcheck(grid, threshold):
    """
    Indique si la grille dont `grid` est la complémentaire est admissible
    dans sa dernière dimension.

    Paramètres :
    ------------
    grid : tableau numpy
        Tableau contenant des 0 (espace occupé) et des 1 (espace libre).
    threshold : scalaire
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
    grid : tableau numpy
        Tableau contenant des 0 (espace libre) et des 1 (espace occupé).
    threshold : scalaire
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
        if res == False:
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
    grid : tableau numpy
        Tableau contenant des 0 (espace libre) et des 1 (espace occupé).
    threshold : scalaire
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
    shape : entier positif, tuple d'entiers positifs
        Dimensions de la grille.
    npoints : entier positif
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

## mole

**mole** est un programme visant à résoudre le problème d'optimisation *Le jardinier et les taupes*, proposé dans le cadre d'un challenge professionnel.

# Un problème de taupes

On considère un jardinier qui possède un jardin rectangulaire découpé en $n \times m$ cases carrées. Ce jardin est régulièrement envahi par des taupes, mesurant  3 cases sur 1, c'est-à-dire :

  - soit 3 cases en longueur et 1 en largeur ;

  - soit 1 case en longueur et 3 en largeur.

Le jardinier cherche à disposer des pièges dans son jardin (chaque piège occupe exactement une case), avec deux contraintes :

  - aucune taupe ne doit pouvoir pénétrer dans le jardin ;

  - il souhaite poser le moins de pièges possibles.

Le placement de certains pièges peut être imposé.

**Exemple :** dans le cas où le jardin est de taille $5 \times 5$, la solution optimale est atteinte pour 8 pièges, et est :

|   |   | X |   |   |
|   |   | X |   |   |
| X | X |   | X | X |
|   |   | X |   |   |
|   |   | X |   |   |


## Licence

Les scripts proposés ici sont sous la licence MIT.

# ReadMe:  Technical Test Cementys

## 1. Exploration des données : 

### 1.1. Lecture et affichage des données :
La lecture des données est réalisé à l'aide de la DataFrame Pandas avec une ligne de code. Une fois la base de donnée récupérée, la phase d'exploration commence. A l'aide de la fonction info() sur la dataframe, un affichage contenant des informations sur les lignes, les colonnes, le nombre des valeurs null ..etc, l'affichage est le suivant:

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1848 entries, 0 to 1847
    Data columns (total 4 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   TIMESTAMP       1848 non-null   object 
     1   deplacement     1848 non-null   object 
     2   ensoleillement  1848 non-null   float64
     3   temperature     1848 non-null   float64
    dtypes: float64(2), object(2)
    memory usage: 57.9+ KB
    None

Nous pouvons donc voir que notre dataset est constitué de 4 colonnes, la première représente la date et l'heure de l'information, la deuxième représente le déplacement enregistré, la troisième l'ensoleillement et la quatrième la température.

On peut remarquer également que tout les valeurs sont différent de NULL, et que la date et le déplacement sont de type object, alors que l'ensoleillement et la temperature sont de type float64.

La deuxième étape consiste à visualiser quelque lignes de la dataset, pour le faire, la fonction head() nous permet de voir un affichage des 5 premières lignes, l'affichage est le suivant: 

    TIMESTAMP deplacement  ensoleillement  temperature
    0  2020-03-14 00:01:22         1.3             0.0     8.381906
    1  2020-03-14 00:18:05         0.5             0.0     8.388235
    2  2020-03-14 00:35:17         1.8             0.0     8.397227
    3  2020-03-14 00:52:39         NAN             0.0     8.411952
    4  2020-03-14 01:09:27         1.7             0.0     8.426197

En voyant les 5 premières lignes de la data, on peux voir une valeur a NAN, qui est NULL, alors qu'elle n'était pas detecté par la fonction info() précédente. Pour vérifier la raison, un affichage de type de cette valeur qui a l'index 3 de déplacement.
Après l'affichage de type, on constate que le type de déplacement est str, donc le NAN est équivalent à écrire "NAN". A la base de cette information, on trouve encore 87 valeurs a "NAN".
 
D'autres vérification sont nécessaire, le type de la date est également str, donc, une vérification de format est très utile, après la vérification à l'aide de module datetime, on constate que toutes les lignes respectent le format.
La vérification suivante concerne l'ensoleillement, ceci ne doit jamais être négatif, et après vérification à l'aide de la fonction loc, on trouvera qu'aucune valeur d'ensoleillement n'est négative.
la température pourra être positive ou négative.
 
L'étape suivante consiste à explorer encore plus les différentes data, et les préparer également à la visualisation. Une première étape est de définir les marges des données, ceci est fait grâce au min et max de chaque colonne. Pandas propose une manipulation facile pour ça avec les fonctions min() et max(). Les résultats sont les suivants :
 
   Le degré minimum de la température est de -0.04473475425
   Le degré maximum de la température est de 27.20630033333333
  
   Le degré minimum de la ensoleillement est de 0.0
   Le degré maximum de l'ensoleillement est de 1378.8043816666666
 
Pour la préparation de la data à la data visualisation pour pouvoir détecter les différentes corrélations, j'ai fais la classification suivante pour la température et l'ensoleillement


- **La temperature :**

|  valeur  |  description  | classe  | Nombre des lignes|
| ------------ | ------------ | ------------ ||
|  1 | Très Faible  |  temp < 5 |238|
|  2 |  Faible | 5 < temp < 10  |740|
|  3 |     Moyenne |10 < temp < 15 |585|
|  4 |  Elevée |  15 < temp < 20 |204|
|  5 |  Très Elevée |    temp > 20 |81|

- **L'ensoleillement : **


|  valeur  |  description  | classe  | Nombre des lignes|
| ------------ | ------------ | ------------ | ------------ |
|  0 |  Très Faible  | ens == 0  | 725  |
|  1 |  Faible |  0 < ens < 100 | 360  |
|  2 |  Moyen | 100 < ens < 300  |  201 |
|  3 |  Elevé | 300 < ens < 500  |  187 |
|  4 |  Très Elevé | 500 < ens < 800  |  155 |
|  5 |  Très très elevé | ens > 800  | 220  |

Ces classe ont pour but de faire un pie chart qui pourra simplifier la corrélation entre l'ensoleillement et la température.

### 1.2. Netoyage des données :

#### 1.2.1. Les valeurs a 'NAN':

Après la phase de l'exploitation des données, on a trouvé que 87 valeurs de déplacement a "NAN", parmi les solutions possible, c'est de remplacer chaque valeur a NAN par une valeur random comprise entre le min et le max de déplacement, ou bien de remplacer tout les valeurs a NAN avec la valeur moyenne de déplacement, mais ces solutions peuvent pénaliser le modèle avec des fausses informations, surtout que le volume de la data utilisé est petit. Donc la meilleure solution est de supprimer tout ligne contenant la valeur 'NAN' comme déplacement parce que notre but est d'avoir le déplacement sans les effets de la nature, et vu que cette information est manquante, ces lignes ne peuvent pas nous aider.


#### 1.2.2. Le type str: 

Une autre remarque qu'on a trouvé lors de l'exploitation des données, c'est bien le type de déplacement, il s'agit d'un type object, qu'on a trouvé après que c'est de str, alors que a la base, il contient des informations numérique. Cette méthode pourra pénaliser également notre modèle, et pour cela, une transformation de type en float64 comme le type de température et ensoleillement est vraiment nécessaire.
Par contre, le type de date en str ne dérange pas, déjà parce que la date et l'heure ne font pas partie des variables qu'on veut découler de la variable déplacement, ainsi que leur type str est acceptable parce qu'ils contient des informations sur la date et non pas numérique.

## 2. La corélation : 

### 2.1. La corélation entre les différentes variables: 

Pour la visualisation de la corrélation entre les différentes variables, l'utilisation de heatmap de seaborn, avec l'utilisation de la matrice de corrélation, ceci donne  un résultat de corrélation entre les différentes variables entre [-1,1] ou -1 est une corrélation parfaite négative, et 1 parfaite positive, le résultat de la dataset est le suivant :
 
[![](https://i.ibb.co/vzxF2NJ/Cor-lation.png)](http://https://i.ibb.co/vzxF2NJ/Cor-lation.png)

### 2.2. La relation entre la temperature et l'ensoleillement: 

Pour voir la relation entre la température et l'ensoleillement, une utilisation des classes déjà définies est nécessaire, ceci est fait aussi grâce au graph pie chart de matplotlib, pour chaque classe de température, une composition des différentes classes de l'ensoleillement, le résultat est le suivant:


[![](https://i.ibb.co/zHGW6vJ/temp-ens.png)](https://i.ibb.co/zHGW6vJ/temp-ens.pnghttp://)

Ou les classes de la température sont par ordre de gauche à droite.

## 3. La réalisation de solution: 

### 3.1. Le choix de l'algorithme: 
Il existe plusieurs algorithmes de machine learning et deep learning qui sont capable de résoudre le problème donnée, les réseaux de neurons sont dans la tête des algorithme optimale, par contre, ils nécessite une grande quantité de données, ce qui n'est pas notre cas. Pour cela, j'ai opté pour une solution plus simple, il s'agit de linear regression.

### 3.2. Une première réalisation : 
Une première réalisation a été faite avec le module Sickit Learn, une transformation des types a été réalisé également, dans le but d'optimiser le modèle de régression.
Les variables choisis sont TIMETAMPS, température et ensoleillement, le résultat est le déplacement.
Une fois notre algorithme est entraîné avec la fonction fit(), une création des information a produire a été nécessaire, la valeur de l'ensoleillement et de la température sont toujours a 0, on essaye donc de faire une prédication pour avoir les valeur de déplacement pour chaque jour en chaque pas de temps sans l'effet de l'ensoleillement et la température.


## 4. Enregisrement des résultat : 

Une fois le résultat de la prédiction est prêt, je le stock directement affecté au TIMESTAMP et aussi l'ancienne valeur de déplacement pour avoir la possibilité de visualiser les changement. Ceci est simplement fait par la création d'une dataframe en passant un dictionnaire qui représente les colonnes et les lignes, et puis sauvegarder cette résultat dans un fichier CSV avec la fonction to_csv().

## 5. Optimisation : 

Après la consultation des résultats, et une autre analyses des données, j'ai pu remarqué qu'on peux mieux utiliser le TIMESTAMP, en effet, y'a une possibilité de la décomposé en date et temps indépendamment.
Le jour pourra etre le meme, donc on peux le laisser comme il l'est, par contre le temps changera de valeur pour chaque seconde, alors qu'il pourra avoir le même effet. Donc le mieux est de décomposer le temps en des périodes, tel que :



|  valeur  |  description  | classe  | 
| ------------ | ------------ | ------------ | 
|  0 |  Nuit  | entre 00:00 et 04:00  |
|  1 |  Deb matin |  entre 04:00 et 07:00 |
|  2 |  Matin | entre 07:00 et 11:00  | 
|  3 |  Midi | entre 11:00 et 15:00  |  
|  4 |  Soir | entre 15:00 et 19:00 |  
|  5 |  Deb nuit | entre 19:00 et 23:59 | 
 
Après les nouvelles modifications, j'aurai le graphe de corrélation suivant: 

[![](https://i.ibb.co/2sFFB9Y/New-Cor.png)](http://https://i.ibb.co/2sFFB9Y/New-Cor.png)







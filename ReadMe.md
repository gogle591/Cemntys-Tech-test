# ReadMe:  Technical Test Cementys

## 1. Exploration des données : 

### 1.1. Lecture et affichage des données :
La lecture des données est réalisé a l'aide de la DataFrame Pandas avec une ligne de code. Une fois la base de donnée recupérée, la phase d'exploration commence. A l'aide de la fonction info() sur la dataframe, un affichage cotenant des informations sur les lignes, les colonnes, le nombre des valeurs null ..ect, l'affichage est le suivant: 

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

Nous pouvons donc voir que notre dataset est constitué de 4 colonnes, la première represente la date et l'heure de l'information, la deuxième représente le deplacement enregestré, la troisème l'ensoleillement et la quatrière la temperature.

On peux remarquer également que tout les valeurs sont different de NULL, et que la date et le deplacemennt sont de type object, alors que l'enseolleiement et la temperature sont de type float64.

La deuxième étape consiste a visualiser quelque lignes de la dataset, pour le faire, la fonction head() nous permet de voir un affichage des 5 premières lignes, l'affichage est le suivant: 

    TIMESTAMP deplacement  ensoleillement  temperature
    0  2020-03-14 00:01:22         1.3             0.0     8.381906
    1  2020-03-14 00:18:05         0.5             0.0     8.388235
    2  2020-03-14 00:35:17         1.8             0.0     8.397227
    3  2020-03-14 00:52:39         NAN             0.0     8.411952
    4  2020-03-14 01:09:27         1.7             0.0     8.426197

En voyant les 5 premières lignes de la data, on peux voir une valeur a NAN, qui est NULL, alors qu'elle n'été pas detecté par la fonction info() précédente. Pour vérifier la raison, un affichage de type de cette valeur qui a l'index 3 de deplacement. 
Après l'affichage de type, on constate que le type de deplacement est str, donc le NAN est équivalant a ecrire "NAN". A la base de cette information, on trouve encore 87 valeurs a "NAN".

D'autres vérification sont nécessaire, le type de la date est également str, donc, une vérification de format est très utile, après la vérification a l'aide de module datetime, on constate que toutes les lignes respectent le format. 
La verficiation suivante concerne l'ensoleillement, ceci ne doit jamais etre négatif, et après vérification a l'aide de la fonction loc, on trouvera qu'aucune valeur d'ensoleillement n'est négative.
la temperature pourra etre positive ou négative.

L'étape suivante consiste a explorer encore plus les différentes data, et les préparer également a la visualisation. Une première étape est de défnir les marges des données, ceci est fait grace au min et max de chaque colonne. Pandas propsoe une manipulation facile pour ça avec les fonctions min() et max(). Les resultats sont les suivants :

    Le degre minimum de la temperature est de -0.04473475425 
    Le degre maximum de la temperature est de 27.20630033333333 
    
    Le degre minimum de la ensoleillement est de 0.0 
    Le degre maximum de la ensoleillement est de 1378.8043816666666

Pour la préparation de la data a la data visualisation pour pouvoir détecter les différentes corélations, j'ai fais la classification suivante pour la temperature et l'enseolleilement 

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

Ces classse ont pour but de faire un pie chart qui pourra simplifier la corélation entre l'ensolleielemnt et la temperature.

### 1.2. Netoyage des données :

#### 1.2.1. Les valeurs a 'NAN':

Après la phase de l'exploitation des données, on a trouvé que 87 valeurs de déplacement a "NAN", parmi les solutions possible, c'est de remplacer chaque valeur a NAN par une valeur random comprise entre le min et le max de deplacement, ou bien de remaplcer tout les valeurs a NAN avec la valeur moyenne de deplacement, mais ces solutions peuvent pénaliser le modèle avec des fausses informations, surtout que le volume de la data utilisé est petit. Donc la meilleure solution est de supprimer tout ligne contenant la valeur 'NAN' comme déplacement parce que notre but est d'avoir le déplacement sans les effets de la nature, et vu que cette information est manquate, ces lignes ne peuvent pas nous aider. 

#### 1.2.2. Le type str: 

Une autre remarque qu'on a trouvé lors de l'exploitation des données, c'est bien le type de deplacement, il s'agit d'un type object, qu'on a trouvé après que c'est de str, alors que a la base, il contient des informations numérique. Cette méthode pourra pénaliser également notre modèle, et pour cela, une transformation de type en float64 comme le type de temperature et ensolleimment est vraiemnt nécessaire.
Par contre, le type de date en str ne dérange pas, déja parce que la date et l'heure ne font pas partie des variables qu'ont veux décoréler de la variable déplacement, ainsi que leur type str est acceptable parce qu'ils contient des informations sur la date et non pas numérique.

## 2. La corélation : 

### 2.1. La corélation entre les différentes variables: 

Pour la visualisation de la corélation entre les différentes variables, l'utilisation de heatmap de seaborn, avec l'utilisation de la matrice de corélation, ceci donne  un résultat de corélation entre les différentes variables entre [-1,1] ou -1 est une corélation parfaite négative, et 1 parfaite positive, le résultat de la dataset est le suivant : 

[![](https://i.ibb.co/vzxF2NJ/Cor-lation.png)](http://https://i.ibb.co/vzxF2NJ/Cor-lation.png)

### 2.2. La relation entre la temperature et l'ensoleillement: 

Pour voir la relation entre la temperature et l'ensolleillement, une utilisation des classes déja définit est nécessaire, ceci est fait aussi grace au graph pie chart de matplotlib, pour chaque classe de temperature, une composition des différentes classes de l'ensoleillement, le resultat est le suivant: 

[![](https://i.ibb.co/zHGW6vJ/temp-ens.png)](https://i.ibb.co/zHGW6vJ/temp-ens.pnghttp://)

Ou les classes de la temperature sont par ordre de gauche a droite.

## 3. La réalisation de solution: 

### 3.1. Le choix de l'algorithme: 
Il existe plusieurs algorithmes de machine learning et deep learning qui sont capable de résoudre le problème donnée, les réseaux de neurons sont dans la tete des algorithle optimale, par contre, ils nécesite une grande quantité de données, ce qui n'est pas notre cas. Pour cela, j'ai opté pour une solution plus simple, il s'agit de linear regression.

### 3.2. Une première réalisation : 
Une première réalisation a été faite avec le module Sickit Learn, une transformation des types a été réalisé également, dans le but d'optimiser le modèle de regression. 
Les variables choisis sont TIMETAMPS, temperature et ensoleillement, le resultat est le déplacement.
Une fois notre algorithme est entrainé avec la fonction fit(), une création des information a produire a été nécessaire, la valeur de l'ensoleillement et de la temperature sont toujours a 0, on essaye donc de faire une prédication pour avoir les valeur de déplacement pour chaque jour en chaque pas de temps sans l'effet de l'ensoleillement et la temperature.










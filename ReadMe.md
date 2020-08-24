# ReadMe:  Technical Test Cementys

## 1. Exploration des donn�es : 

### 1.1. Lecture et affichage des donn�es :
La lecture des donn�es est r�alis� a l'aide de la DataFrame Pandas avec une ligne de code. Une fois la base de donn�e recup�r�e, la phase d'exploration commence. A l'aide de la fonction info() sur la dataframe, un affichage cotenant des informations sur les lignes, les colonnes, le nombre des valeurs null ..ect, l'affichage est le suivant: 

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

Nous pouvons donc voir que notre dataset est constitu� de 4 colonnes, la premi�re represente la date et l'heure de l'information, la deuxi�me repr�sente le deplacement enregestr�, la trois�me l'ensoleillement et la quatri�re la temperature.

On peux remarquer �galement que tout les valeurs sont different de NULL, et que la date et le deplacemennt sont de type object, alors que l'enseolleiement et la temperature sont de type float64.

La deuxi�me �tape consiste a visualiser quelque lignes de la dataset, pour le faire, la fonction head() nous permet de voir un affichage des 5 premi�res lignes, l'affichage est le suivant: 

    TIMESTAMP deplacement  ensoleillement  temperature
    0  2020-03-14 00:01:22         1.3             0.0     8.381906
    1  2020-03-14 00:18:05         0.5             0.0     8.388235
    2  2020-03-14 00:35:17         1.8             0.0     8.397227
    3  2020-03-14 00:52:39         NAN             0.0     8.411952
    4  2020-03-14 01:09:27         1.7             0.0     8.426197

En voyant les 5 premi�res lignes de la data, on peux voir une valeur a NAN, qui est NULL, alors qu'elle n'�t� pas detect� par la fonction info() pr�c�dente. Pour v�rifier la raison, un affichage de type de cette valeur qui a l'index 3 de deplacement. 
Apr�s l'affichage de type, on constate que le type de deplacement est str, donc le NAN est �quivalant a ecrire "NAN". A la base de cette information, on trouve encore 87 valeurs a "NAN".

D'autres v�rification sont n�cessaire, le type de la date est �galement str, donc, une v�rification de format est tr�s utile, apr�s la v�rification a l'aide de module datetime, on constate que toutes les lignes respectent le format. 
La verficiation suivante concerne l'ensoleillement, ceci ne doit jamais etre n�gatif, et apr�s v�rification a l'aide de la fonction loc, on trouvera qu'aucune valeur d'ensoleillement n'est n�gative.
la temperature pourra etre positive ou n�gative.

L'�tape suivante consiste a explorer encore plus les diff�rentes data, et les pr�parer �galement a la visualisation. Une premi�re �tape est de d�fnir les marges des donn�es, ceci est fait grace au min et max de chaque colonne. Pandas propsoe une manipulation facile pour �a avec les fonctions min() et max(). Les resultats sont les suivants :

    Le degre minimum de la temperature est de -0.04473475425 
    Le degre maximum de la temperature est de 27.20630033333333 
    
    Le degre minimum de la ensoleillement est de 0.0 
    Le degre maximum de la ensoleillement est de 1378.8043816666666

Pour la pr�paration de la data a la data visualisation pour pouvoir d�tecter les diff�rentes cor�lations, j'ai fais la classification suivante pour la temperature et l'enseolleilement 

- **La temperature :**

|  valeur  |  description  | classe  | Nombre des lignes|
| ------------ | ------------ | ------------ ||
|  1 | Tr�s Faible  |  temp < 5 |238|
|  2 |  Faible | 5 < temp < 10  |740|
|  3 |     Moyenne |10 < temp < 15 |585|
|  4 |  Elev�e |  15 < temp < 20 |204|
|  5 |  Tr�s Elev�e |    temp > 20 |81|

- **L'ensoleillement : **


|  valeur  |  description  | classe  | Nombre des lignes|
| ------------ | ------------ | ------------ | ------------ |
|  0 |  Tr�s Faible  | ens == 0  | 725  |
|  1 |  Faible |  0 < ens < 100 | 360  |
|  2 |  Moyen | 100 < ens < 300  |  201 |
|  3 |  Elev� | 300 < ens < 500  |  187 |
|  4 |  Tr�s Elev� | 500 < ens < 800  |  155 |
|  5 |  Tr�s tr�s elev� | ens > 800  | 220  |

Ces classse ont pour but de faire un pie chart qui pourra simplifier la cor�lation entre l'ensolleielemnt et la temperature.

### 1.2. Netoyage des donn�es :

#### 1.2.1. Les valeurs a 'NAN':

Apr�s la phase de l'exploitation des donn�es, on a trouv� que 87 valeurs de d�placement a "NAN", parmi les solutions possible, c'est de remplacer chaque valeur a NAN par une valeur random comprise entre le min et le max de deplacement, ou bien de remaplcer tout les valeurs a NAN avec la valeur moyenne de deplacement, mais ces solutions peuvent p�naliser le mod�le avec des fausses informations, surtout que le volume de la data utilis� est petit. Donc la meilleure solution est de supprimer tout ligne contenant la valeur 'NAN' comme d�placement parce que notre but est d'avoir le d�placement sans les effets de la nature, et vu que cette information est manquate, ces lignes ne peuvent pas nous aider. 

#### 1.2.2. Le type str: 

Une autre remarque qu'on a trouv� lors de l'exploitation des donn�es, c'est bien le type de deplacement, il s'agit d'un type object, qu'on a trouv� apr�s que c'est de str, alors que a la base, il contient des informations num�rique. Cette m�thode pourra p�naliser �galement notre mod�le, et pour cela, une transformation de type en float64 comme le type de temperature et ensolleimment est vraiemnt n�cessaire.
Par contre, le type de date en str ne d�range pas, d�ja parce que la date et l'heure ne font pas partie des variables qu'ont veux d�cor�ler de la variable d�placement, ainsi que leur type str est acceptable parce qu'ils contient des informations sur la date et non pas num�rique.

## 2. La cor�lation : 

### 2.1. La cor�lation entre les diff�rentes variables: 

Pour la visualisation de la cor�lation entre les diff�rentes variables, l'utilisation de heatmap de seaborn, avec l'utilisation de la matrice de cor�lation, ceci donne  un r�sultat de cor�lation entre les diff�rentes variables entre [-1,1] ou -1 est une cor�lation parfaite n�gative, et 1 parfaite positive, le r�sultat de la dataset est le suivant : 

[![](https://i.ibb.co/vzxF2NJ/Cor-lation.png)](http://https://i.ibb.co/vzxF2NJ/Cor-lation.png)

### 2.2. La relation entre la temperature et l'ensoleillement: 

Pour voir la relation entre la temperature et l'ensolleillement, une utilisation des classes d�ja d�finit est n�cessaire, ceci est fait aussi grace au graph pie chart de matplotlib, pour chaque classe de temperature, une composition des diff�rentes classes de l'ensoleillement, le resultat est le suivant: 

[![](https://i.ibb.co/zHGW6vJ/temp-ens.png)](https://i.ibb.co/zHGW6vJ/temp-ens.pnghttp://)

Ou les classes de la temperature sont par ordre de gauche a droite.

## 3. La r�alisation de solution: 

### 3.1. Le choix de l'algorithme: 
Il existe plusieurs algorithmes de machine learning et deep learning qui sont capable de r�soudre le probl�me donn�e, les r�seaux de neurons sont dans la tete des algorithle optimale, par contre, ils n�cesite une grande quantit� de donn�es, ce qui n'est pas notre cas. Pour cela, j'ai opt� pour une solution plus simple, il s'agit de linear regression.

### 3.2. Une premi�re r�alisation : 
Une premi�re r�alisation a �t� faite avec le module Sickit Learn, une transformation des types a �t� r�alis� �galement, dans le but d'optimiser le mod�le de regression. 
Les variables choisis sont TIMETAMPS, temperature et ensoleillement, le resultat est le d�placement.
Une fois notre algorithme est entrain� avec la fonction fit(), une cr�ation des information a produire a �t� n�cessaire, la valeur de l'ensoleillement et de la temperature sont toujours a 0, on essaye donc de faire une pr�dication pour avoir les valeur de d�placement pour chaque jour en chaque pas de temps sans l'effet de l'ensoleillement et la temperature.










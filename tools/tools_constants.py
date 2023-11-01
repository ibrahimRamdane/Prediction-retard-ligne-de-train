"""
Python module containing the main constants of the code.

Constants
---------
PATH_DATASET : str
    Path to the dataset of the TGV delays.

PATH_FIGURES : str
    Path to the saved figures folder

DELAY_FEATURE : str
    Name of the feature representing the mean delay of the TGV at arrival.

LIST_CAUSE_FEATURES : list[str]
    List containing the name of the features representing the different causes of delay.

LIST_FEATURES : list[str]
    List containing all features

ADDED_COL : list[str]
    List containing our added columns in the dataset

QUANT_FEATURES : list[str]
    List of column to normalize

DROPPED_COLS : list[str]
    List contaning the column dropped for training

RANDOM_STATE : int
    Random seed used to train the models.
"""

#################
### Constants ###
#################

### Paths ###

PATH_DATASET = "Data/TGV.csv"
PATH_FIGURES = "figures/"

### Features ###

DELAY_FEATURE = "retard_moyen_arrivee"
LIST_CAUSE_FEATURES = [
    "prct_cause_externe",
    "prct_cause_infra",
    "prct_cause_gestion_trafic",
    "prct_cause_materiel_roulant",
    "prct_cause_prise_en_charge_voyageurs",
]

LIST_FEATURES = [
    "duree_moyenne",
    "gare_depart",
    "gare_arrivee",
    "service",
    "nb_train_prevu",
    "nb_annulation",
    "commentaire_annulation",
    "nb_train_depart_retard",
    "retard_moyen_tous_trains_depart",
    "commentaire_retards_depart",
    "commentaires_retard_arrivee",
    "nb_train_retard_arrivee",
    "retard_moyen_trains_retard_sup15",
    "nb_train_retard_sup_15",
    "nb_train_retard_sup_30",
    "nb_train_retard_sup_60",
    "date",
    "retard_moyen_arrivee",
    "retard_moyen_depart"
]

ADDED_COL = [
    "gare_depart_coord_x",
    "gare_depart_coord_y",
    "gare_arrivee_coord_x",
    "gare_arrivee_coord_y"
]

LIST_ALL_POSSIBLE_FEATURES = LIST_FEATURES #+ ADDED_COL

QUANT_FEATURES = [
    "duree_moyenne",
    "nb_train_prevu",
    "retard_moyen_depart"
]

CAT_FEATURES = [
    "service",
    "gare_depart",
    "gare_arrivee"
]

DROPPED_COLS = [
    "commentaire_annulation",
    "commentaire_retards_depart",
    "commentaires_retard_arrivee",
    "retard_moyen_arrivee",
    "nb_train_prevu",
    "nb_annulation", 
    "retard_moyen_trains_retard_sup15",
    "nb_train_retard_sup_15",
    "nb_train_retard_sup_30",
    "nb_train_retard_sup_60",
    "retard_moyen_tous_trains_depart",
    "nb_train_depart_retard", 
    "retard_moyen_depart", 
    "nb_train_retard_arrivee"
]

# FEATURES_TO_PASS_COORD = [x for x in LIST_ALL_POSSIBLE_FEATURES if (
#     x in QUANT_FEATURES) == False and (x in DROPPED_COLS) == False
#     and (x in CAT_FEATURES) == False]


# FEATURES_TO_PASS_BINARY = [x for x in LIST_ALL_POSSIBLE_FEATURES if (
#     x in QUANT_FEATURES) == False and (x in DROPPED_COLS) == False
#     and (x in CAT_FEATURES) == False and (x in ADDED_COL) == False]

FEATURES_TO_PASS_COORD = [x for x in LIST_ALL_POSSIBLE_FEATURES if  (not(x in DROPPED_COLS) and not(x in ADDED_COL) and not(x in CAT_FEATURES))]


FEATURES_TO_PASS_BINARY = [x for x in LIST_ALL_POSSIBLE_FEATURES if  (not(x in DROPPED_COLS) and not(x in ADDED_COL) and not(x in CAT_FEATURES))]

### Others ###

TEST_MODE = False

RANDOM_STATE = 42

ALPH = 1.0
TOLERANCE = 0.0001
ITER_MAX = 1000
L1_RATIO = 0.5

from pulp import *
import pandas as pd

INGREDIENTS_DATA = {
    "Riz Blanc":   {"Prix_Unitaire": 0.20, "Calories": 350, "Protéines": 6.0, "Fibres": 1.0, "Sel": 0.0},
    "Poulet":      {"Prix_Unitaire": 1.20, "Calories": 165, "Protéines": 31.0, "Fibres": 0.0, "Sel": 0.1},
    "Lentilles":   {"Prix_Unitaire": 0.35, "Calories": 116, "Protéines": 9.0, "Fibres": 8.0, "Sel": 0.0},
    "Carottes":    {"Prix_Unitaire": 0.10, "Calories": 41, "Protéines": 0.9, "Fibres": 2.8, "Sel": 0.05},
    "Épices":      {"Prix_Unitaire": 5.00, "Calories": 300, "Protéines": 5.0, "Fibres": 10.0, "Sel": 1.0},
}

BUDGET_MAX_PLAT = 4.00
PORTIONS = 1
POIDS_TOTAL_MIN = 500
POIDS_TOTAL_MAX = 80

# Contraintes Nutritionnelles pour le PLAT ENTIER
CALORIES_RECETTE_MIN = 800      # kcal
PROTEINES_RECETTE_MIN = 60      # grammes
SEL_RECETTE_MAX = 1.0           # grammes (faible pour une recette de 1 portions)

# Le "Score de Qualité" (Fonction Objectif) : Poids pour chaque nutriment (pour 100g)
# On maximise Protéines et Fibres, et on pénalise fortement le Sel.
QUALITE_POIDS = {"Protéines": 1.0, "Fibres": 1.5, "Sel": -5.0}

qualite_scores = {}
for ingredient, data in INGREDIENTS_DATA.items():
    score = (data["Protéines"] * QUALITE_POIDS["Protéines"] +
             data["Fibres"] * QUALITE_POIDS["Fibres"] +
             data["Sel"] * QUALITE_POIDS["Sel"])
    qualite_scores[ingredient] = score


# --- 2. CRÉATION DU MODÈLE ET DES VARIABLES ---

# Problème de maximisation de la qualité
probleme = LpProblem("Optimisation_Recette_Journalière", LpMaximize)

# Variables de décision: x[i] est la quantité d'ingrédient i (en unités de 100g)
x = LpVariable.dicts("Qte_100g", INGREDIENTS_DATA.keys(), lowBound=0)


# --- 3. FONCTION OBJECTIF (MAXIMISATION DE LA QUALITÉ) ---

# Maximiser la somme des scores de qualité (pondérés par les quantités en 100g)
probleme += lpSum(qualite_scores[i] * x[i] for i in INGREDIENTS_DATA.keys()), "Score de Qualité de la Recette"


# --- 4. CONTRAINTES ---

# A. Contrainte Budgétaire 💰
probleme += lpSum(INGREDIENTS_DATA[i]["Prix_Unitaire"] * x[i] for i in INGREDIENTS_DATA.keys()) <= BUDGET_MAX_PLAT, "Budget Max du Plat"

# B. Contraintes de Volume/Poids de la Recette (x[i] est en 100g)
probleme += lpSum(x) * 100 >= POIDS_TOTAL_MIN, "Poids Min Total (g)"
probleme += lpSum(x) * 100 <= POIDS_TOTAL_MAX, "Poids Max Total (g)"

# C. Contraintes Nutritionnelles
probleme += lpSum(INGREDIENTS_DATA[i]["Calories"] * x[i] for i in INGREDIENTS_DATA.keys()) >= CALORIES_RECETTE_MIN, "Calories Min Recette"
probleme += lpSum(INGREDIENTS_DATA[i]["Protéines"] * x[i] for i in INGREDIENTS_DATA.keys()) >= PROTEINES_RECETTE_MIN, "Protéines Min Recette"
probleme += lpSum(INGREDIENTS_DATA[i]["Sel"] * x[i] for i in INGREDIENTS_DATA.keys()) <= SEL_RECETTE_MAX, "Sel Max Recette"

# D. Contraintes de Faisabilité et de Goût (pour garantir la variété)
probleme += x["Carottes"] >= 1.0, "Min Carottes (100g)"     # Au moins 100g de carottes
probleme += x["Poulet"] >= 1.5, "Min Poulet (100g)"         # Au moins 150g de poulet (pour 2)
probleme += x["Épices"] <= 0.2, "Max Epices (100g)"         # Max 20g d'épices (quantité réaliste)
probleme += x["Riz Blanc"] <= 3.0, "Max Riz (100g)"         # Max 300g de riz

# --- 5. RÉSOUDRE LE PROBLÈME ---

probleme.solve()

# --- 6. AFFICHAGE DES RÉSULTATS DÉTAILLÉS ---

print("=" * 50)
print(f"ALGORITHME D'OPTIMISATION DE RECETTE (2 PORTIONS)")
print(f"Statut de la résolution : {LpStatus[probleme.status]}")
print(f"Score de Qualité Maximale Atteint : {value(probleme.objective):.2f}")
print("=" * 50)
print(f"CONTRAINTES : Budget Max {BUDGET_MAX_PLAT:.2f} € | Min Protéines {PROTEINES_RECETTE_MIN} g")
print("-" * 50)
print("RECETTE OPTIMALE (Quantités pour 2 Portions) :")
print("-" * 50)

results = []
total_cout = 0
total_calories = 0
total_poids = 0
total_proteines = 0
total_fibres = 0
total_sel = 0

for v in probleme.variables():
    if v.varValue > 0.001:
        quantite_unit = v.varValue
        quantite_g = quantite_unit * 100
        aliment = v.name.replace("Qte_100g_", "").replace("_", " ")
        data = INGREDIENTS_DATA[aliment]

        cout = quantite_unit * data["Prix_Unitaire"]
        calories = quantite_unit * data["Calories"]
        proteines = quantite_unit * data["Protéines"]
        fibres = quantite_unit * data["Fibres"]
        sel = quantite_unit * data["Sel"]

        results.append({
            "Ingrédient": aliment,
            "Quantité (g)": f"{quantite_g:.0f}",
            "Coût (€)": f"{cout:.2f}",
            "Protéines (g)": f"{proteines:.1f}",
            "Fibres (g)": f"{fibres:.1f}",
            "Sel (g)": f"{sel:.2f}"
        })

        total_cout += cout
        total_calories += calories
        total_poids += quantite_g
        total_proteines += proteines
        total_fibres += fibres
        total_sel += sel

# Affichage des résultats dans un tableau
df = pd.DataFrame(results).set_index("Ingrédient")
print(df)
print("-" * 50)

print("Bilan Nutritionnel et Coût du Plat Final :")
print(f"  💰 Coût Total : {total_cout:.2f} € (Coût/portion: {total_cout / PORTIONS:.2f} €)")
print(f"  ⚖️ Poids Total : {total_poids:.0f} g")
print(f"  🔥 Calories Totales : {total_calories:.0f} kcal (Cible: {CALORIES_RECETTE_MIN}+)")
print(f"  💪 Protéines Totales : {total_proteines:.1f} g (Cible: {PROTEINES_RECETTE_MIN}+)")
print(f"  🌱 Fibres Totales : {total_fibres:.1f} g")
print(f"  🧂 Sel Total : {total_sel:.2f} g (Max: {SEL_RECETTE_MAX})")
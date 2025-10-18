import pandas as pd
from pulp import *
from sklearn.neighbors import NearestNeighbors
import numpy as np

# ====================================================================
# MODULE 1 : BASE DE CONNAISSANCES & PRÉPARATION DES DONNÉES
# ====================================================================

# Données simplifiées des producteurs (le marché)
producteurs_df = pd.DataFrame({
    'Producteur_ID': [1, 2, 3, 4, 5],
    'Region': ['Antananarivo', 'Toamasina', 'Antananarivo', 'Fianarantsoa', 'Toamasina'],
    'Produit': ['Riz', 'Poisson', 'Zebu', 'Haricots', 'Riz'],
    'Quantite_disponible_kg': [500, 50, 200, 100, 400],
    'Prix_unitaire_euro_kg': [0.5, 3.5, 4.0, 1.2, 0.6]
})

# Données des recettes malgaches (simplifiées)
RECETTES_DATA = {
    "Ravitoto": {
        "Ingredients": {"Feuilles de Manioc": 0.5, "Zebu": 0.3}, # Quantité par portion (kg)
        "Nutriments_par_portion": {"Calories": 500, "Proteines": 40, "Fibres": 15, "Sel": 0.5},
        "Score_Qualite_Base": 8.5 # Score pré-calculé (goût, tradition, etc.)
    },
    "Vary Amin'anana": {
        "Ingredients": {"Riz": 0.2, "Brèdes": 0.1},
        "Nutriments_par_portion": {"Calories": 350, "Proteines": 10, "Fibres": 5, "Sel": 0.2},
        "Score_Qualite_Base": 7.0
    },
    "Lasary Voatabia": {
        "Ingredients": {"Tomates": 0.1, "Oignons": 0.05},
        "Nutriments_par_portion": {"Calories": 150, "Proteines": 3, "Fibres": 3, "Sel": 0.1},
        "Score_Qualite_Base": 6.5
    }
}

# Fonction pour obtenir le prix régional moyen d'un ingrédient
def get_regional_price(region, ingredient):
    """Calcule le prix moyen d'un ingrédient dans une région donnée."""
    prices = producteurs_df[
        (producteurs_df['Region'] == region) & 
        (producteurs_df['Produit'] == ingredient)
    ]['Prix_unitaire_euro_kg']
    return prices.mean() if not prices.empty else None

# ====================================================================
# MODULE 2 : MOTEUR D'OPTIMISATION (PuLP) - L'ALGORYTHME DE MAXIMISATION
# ====================================================================

def optimiser_panier_pour_recette(recette_nom, budget_max_journalier, user_region):
    """
    Optimise le coût des ingrédients d'une recette selon la disponibilité régionale.
    Retourne le coût total et si la recette est faisable.
    """
    data_recette = RECETTES_DATA[recette_nom]
    probleme = LpProblem(f"Optimisation_{recette_nom}", LpMinimize)
    
    ingr_vars = {}
    ingr_prix = {}
    
    # Créer les variables et calculer le prix localisé
    for ingr, qte_necessaire in data_recette["Ingredients"].items():
        prix = get_regional_price(user_region, ingr)
        
        # Si le prix n'est pas trouvé (ingrédient non disponible), la recette est infaisable
        if prix is None:
            return float('inf'), False 
            
        ingr_vars[ingr] = LpVariable(f"Achat_{ingr}", lowBound=qte_necessaire, upBound=qte_necessaire)
        ingr_prix[ingr] = prix

    # Fonction Objectif: Minimiser le Coût Total
    probleme += lpSum(ingr_prix[i] * ingr_vars[i] for i in ingr_vars), "Cout Total de la Recette"

    # Contrainte Budgétaire (Journalière)
    probleme += lpSum(ingr_prix[i] * ingr_vars[i] for i in ingr_vars) <= budget_max_journalier, "Budget Max"

    # Résoudre le problème
    probleme.solve()

    if LpStatus[probleme.status] == "Optimal":
        return value(probleme.objective), True
    else:
        return float('inf'), False # Infaisable ou budget trop serré

# ====================================================================
# MODULE 3 : MOTEUR DE SUGGESTION & PRÉDICTION (IA)
# ====================================================================

def generer_suggestion_ia(budget_journalier, region_utilisateur):
    """
    Prédit les meilleures options et suggère les recettes de meilleure qualité.
    """
    resultats_optimises = []
    
    # 1. Optimisation pour chaque recette
    for nom_recette in RECETTES_DATA:
        # Tente d'optimiser le panier pour cette recette
        cout_optimal, faisable = optimiser_panier_pour_recette(
            nom_recette, budget_journalier, region_utilisateur
        )
        
        if faisable:
            # 2. Calcul du Score Final de Qualité (IA/Scoring)
            # Le score final combine la qualité de base de la recette et une prime/pénalité
            # basée sur le rapport Qualité/Coût.
            
            score_base = RECETTES_DATA[nom_recette]['Score_Qualite_Base']
            
            # Pénalité si le coût est proche du budget, prime s'il est très bas
            budget_utilise_ratio = cout_optimal / budget_journalier
            
            # Fonction de scoring simple : Score Final = Score Base * (1 + (1 - budget_utilise_ratio))
            # Moins on utilise de budget, plus le facteur multiplicatif est grand.
            score_ajuste = score_base * (1 + (1 - budget_utilise_ratio))
            
            # On pourrait aussi intégrer un modèle de ML pour prédire la popularité
            # ou la qualité perçue dans la région. (Simulation ici)
            
            resultats_optimises.append({
                "Recette": nom_recette,
                "Cout_Optimal": cout_optimal,
                "Score_Final_Qualite": round(score_ajuste, 2)
            })

    if not resultats_optimises:
        return "Aucune recette malgache faisable n'a été trouvée pour ce budget/cette région."

    # 3. Suggestion Finale (Trier par Score Final de Qualité)
    suggestions_df = pd.DataFrame(resultats_optimises)
    suggestions_df = suggestions_df.sort_values(by="Score_Final_Qualite", ascending=False).reset_index(drop=True)
    
    return suggestions_df

# ====================================================================
# EXÉCUTION DU PROGRAMME
# ====================================================================

# Paramètres Utilisateur
BUDGET_UTILISATEUR = 5.0 # Euros par jour
REGION_UTILISATEUR = 'Antananarivo'

print(f"Recherche de recettes pour : {REGION_UTILISATEUR} avec un budget de {BUDGET_UTILISATEUR} €/jour.\n")

suggestions = generer_suggestion_ia(BUDGET_UTILISATEUR, REGION_UTILISATEUR)

print("--- RÉSULTATS DE L'IA ---")
if isinstance(suggestions, str):
    print(suggestions)
else:
    print(suggestions)
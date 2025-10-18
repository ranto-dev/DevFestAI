import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import ast

# === Chargement des datasets ===
df_producers = pd.read_csv("producteurs.csv")
df_recipes = pd.read_csv("recettes.csv")

# Convertir les colonnes JSON string -> dictionnaire Python
df_recipes["ingredients_dict"] = df_recipes["ingredients_dict"].apply(ast.literal_eval)

# === Entraînement du modèle IA (avec meilleure performance) ===
training_data = []
for _, recipe in df_recipes.iterrows():
    total_cost = 0
    for ingredient, qty in recipe["ingredients_dict"].items():
        producers = df_producers[df_producers["ingredient_id"] == ingredient]
        if len(producers) > 0:
            avg_price = producers["price"].mean()
            total_cost += avg_price * qty
    training_data.append({
        "budget": total_cost * 1.2,
        "cost": total_cost,
        "region_match": 1,
        "score": recipe["expected_score"]
    })

CURRENCY = "Ariary"

df_train = pd.DataFrame(training_data)

# Normalisation des données pour éviter des valeurs trop grandes
df_train["budget_norm"] = df_train["budget"] / 10000
df_train["cost_norm"] = df_train["cost"] / 10000
df_train["region_match"] = df_train["region_match"]

X = df_train[["budget_norm", "cost_norm", "region_match"]]
y = df_train["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=10)
model.fit(X_train, y_train)

print(f"✅ Données importées :\n- {len(df_producers)} producteurs\n- {len(df_recipes)} recettes")
print(f"📈 Performance du modèle R² : {model.score(X_test, y_test):.2f}\n")

# === Fonction IA : prédire score et coût pour une recette ===
def predict_recipe_score(ingredients_dict, budget, df_producers):
    total_cost = 0
    for ingredient, qty in ingredients_dict.items():
        producers = df_producers[df_producers["ingredient_id"] == ingredient]
        if not producers.empty:
            avg_price = producers["price"].mean()
            total_cost += avg_price * qty
    region_match = 1
    score_pred = model.predict([[budget, total_cost, region_match]])[0]
    return score_pred, total_cost

# === Système de suggestion ===
def suggest_recipes(region, budget):
    print(f"💡 Suggestions pour la région {region} avec un budget de {budget:,.0f} {CURRENCY} :\n")
    suggestions = []

    for _, rec in df_recipes.iterrows():
        pred, cost = predict_recipe_score(rec["ingredients_dict"], budget, df_producers)
        if cost <= budget * 1.2:
            suggestions.append((rec["recipe_name"], round(pred, 3), round(cost, 0)))

    suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)

    for name, score, cost in suggestions[:5]:
        print(f"- {name} → Score prédit: {score}, Coût estimé: {cost:,.0f} {CURRENCY}")


# === Exemple d’utilisation ===
# Exemple d’utilisation (budget de 10 000 Ariary)
suggest_recipes(region="Analamanga", budget=10000)
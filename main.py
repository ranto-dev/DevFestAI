import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import ast
import json

CURRENCY = "Ariary"

df_producers = pd.read_csv("producteurs.csv")
df_recipes = pd.read_csv("recettes.csv")
df_recipes["ingredients_dict"] = df_recipes["ingredients_dict"].apply(ast.literal_eval)

nutrition_data = {
    "riz": {"proteines": 2.5, "fer": 0.3, "vitamineC": 0, "fibres": 0.4, "lipides": 0.2, "score_nutrition": 0.65},
    "poulet": {"proteines": 18, "fer": 1, "vitamineC": 0, "fibres": 0, "lipides": 5, "score_nutrition": 0.85},
    "poisson": {"proteines": 20, "fer": 1.2, "vitamineC": 0, "fibres": 0, "lipides": 4, "score_nutrition": 0.9},
    "légumes": {"proteines": 3, "fer": 0.5, "vitamineC": 30, "fibres": 2, "lipides": 0.5, "score_nutrition": 0.88},
    "huile": {"proteines": 0, "fer": 0, "vitamineC": 0, "fibres": 0, "lipides": 10, "score_nutrition": 0.6},
    "oignon": {"proteines": 1, "fer": 0.1, "vitamineC": 7, "fibres": 1, "lipides": 0.2, "score_nutrition": 0.7},
    "sucre": {"proteines": 0, "fer": 0, "vitamineC": 0, "fibres": 0, "lipides": 0, "score_nutrition": 0.3},
    "viande": {"proteines": 22, "fer": 2, "vitamineC": 0, "fibres": 0, "lipides": 6, "score_nutrition": 0.9},
}

def compute_cost(ingredients_dict):
    total_cost = 0
    for ing, qty in ingredients_dict.items():
        producers = df_producers[df_producers["ingredient_id"] == ing]
        if not producers.empty:
            avg_price = producers["price"].mean()
            total_cost += avg_price * qty
    return total_cost

def compute_nutrition_score(ingredients_dict):
    total_score = 0
    for ing, qty in ingredients_dict.items():
        if ing in nutrition_data:
            total_score += nutrition_data[ing]["score_nutrition"] * qty
    return total_score

training_data = []
for _, rec in df_recipes.iterrows():
    cost = compute_cost(rec["ingredients_dict"])
    nutrition_score = compute_nutrition_score(rec["ingredients_dict"])
    training_data.append({
        "budget": cost * 1.2,
        "cost": cost,
        "nutrition": nutrition_score,
        "region_match": 1,
        "score": rec["expected_score"]
    })

df_train = pd.DataFrame(training_data)

df_train["budget_norm"] = df_train["budget"] / 10000
df_train["cost_norm"] = df_train["cost"] / 10000

X = df_train[["budget_norm", "cost_norm", "nutrition", "region_match"]]
y = df_train["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=10)
model.fit(X_train, y_train)

print(f"Données importées :")
print(f"- {len(df_producers)} producteurs")
print(f"- {len(df_recipes)} recettes")
print(f"Performance du modèle R² : {model.score(X_test, y_test):.2f}\n")

def predict_recipe(ingredients_dict, budget):
    cost = compute_cost(ingredients_dict)
    nutrition = compute_nutrition_score(ingredients_dict)
    pred = model.predict([[budget / 10000, cost / 10000, nutrition, 1]])[0]
    return pred, cost, nutrition

def suggest_recipes(region, budget):
    print(f"Meilleures suggestions pour la région {region} avec un budget de {budget:,.0f} {CURRENCY} :\n")
    suggestions = []

    for _, rec in df_recipes.iterrows():
        pred, cost, nutrition = predict_recipe(rec["ingredients_dict"], budget)
        if cost <= budget * 1.2:
            suggestions.append({
                "name": rec["recipe_name"],
                "pred_score": round(pred, 3),
                "cost": round(cost, 0),
                "nutrition": round(nutrition, 3),
                "ingredients": rec["ingredients_dict"],
            })

    suggestions = sorted(suggestions, key=lambda x: (x["nutrition"], -x["cost"]), reverse=True)

    for i, s in enumerate(suggestions[:5], 1):
        print(f"{i}. {s['name']}")
        print(f"   - Score IA       : {s['pred_score']}")
        print(f"   - Coût estimé    : {s['cost']:,.0f} {CURRENCY}")
        print(f"   - Nutrition index: {s['nutrition']:.2f}")
        print(f"   - Ingrédients    : {s['ingredients']}\n")

    best = suggestions[0]
    best_data = {
        "nom_recette": best["name"],
        "region": region,
        "cout": best["cost"],
        "monnaie": CURRENCY,
        "nutrition_index": best["nutrition"],
        "ingredients": best["ingredients"],
        "score_pred": best["pred_score"]
    }

    # with open("resultat_meilleur.json", "w", encoding="utf-8") as f:
    #     json.dump(best_data, f, ensure_ascii=False, indent=4)

    # print("Meilleure recette exportée sous forme JSON :\n")
    # print(json.dumps(best_data, ensure_ascii=False, indent=4))

    print(f"La meilleur recette trouvée: {best_data}")

suggest_recipes(region="Analamanga", budget=1500)

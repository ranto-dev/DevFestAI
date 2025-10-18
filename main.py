# suggestion_recettes.py
import random
import math
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# pour l'allocation d'achat (optimisation linéaire)
import pulp

# -------------------------
# 1. Génération de données synthétiques (prototype)
# -------------------------
random.seed(0)
np.random.seed(0)

# ingrédients catalogue (quelques ingrédients courants à Madagascar)
ingredients = [
    {"ingredient_id": "riz", "unite": "kg", "regions": ["Analamanga","Bongolava","Itasy"]},
    {"ingredient_id": "poisson", "unite": "kg", "regions": ["Toamasina","Boeny","Analanjirofo"]},
    {"ingredient_id": "légumes", "unite": "kg", "regions": ["Analamanga","Haute-Matsiatra","Itasy"]},
    {"ingredient_id": "poulet", "unite": "kg", "regions": ["Analamanga","Alaotra","Vakinankaratra"]},
    {"ingredient_id": "oignon", "unite": "kg", "regions": ["Analamanga","Itasy","Vakinankaratra"]},
    {"ingredient_id": "huile", "unite": "litre", "regions": ["nationwide"]},
]

df_ingredients = pd.DataFrame(ingredients)

# Générer des producteurs
producers = []
producer_id = 1
regions_all = ["Analamanga","Toamasina","Itasy","Bongolava","Boeny","Alaotra","Haute-Matsiatra","Analanjirofo","Vakinankaratra"]
for ing in df_ingredients['ingredient_id']:
    # 3 à 6 producteurs par ingrédient
    for _ in range(random.randint(3,6)):
        region = random.choice(regions_all)
        price = round(random.uniform(0.5, 5.0), 2)  # prix unitaire arbitraire (ex: X MGA / unité)
        qty = round(random.uniform(10, 200), 1)     # kg ou litres disponibles
        quality = round(random.uniform(0.5, 1.0), 2)  # 0..1
        producers.append({
            "producer_id": f"P{producer_id}",
            "ingredient_id": ing,
            "region": region,
            "price": price,
            "qty": qty,
            "quality": quality
        })
        producer_id += 1
df_producers = pd.DataFrame(producers)

# Générer recettes de base (quelques recettes traditionnelles simplifiées)
recipes = [
    {"id": "R1", "name": "Riz au poisson", "ingredients": {"riz":0.2, "poisson":0.2, "oignon":0.05, "huile":0.02}, "time":30},
    {"id": "R2", "name": "Ravitoto (avec viande)", "ingredients": {"riz":0.2, "légumes":0.25, "poulet":0.15, "oignon":0.05, "huile":0.02}, "time":45},
    {"id": "R3", "name": "Riz sauté légumes", "ingredients": {"riz":0.2, "légumes":0.3, "oignon":0.05, "huile":0.02}, "time":25},
    {"id": "R4", "name": "Poisson grillé + riz", "ingredients": {"riz":0.2, "poisson":0.25, "oignon":0.03, "huile":0.02}, "time":35},
    {"id": "R5", "name": "Poulet au riz", "ingredients": {"riz":0.2, "poulet":0.25, "oignon":0.04, "huile":0.03}, "time":50},
]

df_recipes = pd.DataFrame(recipes)

# -------------------------
# 2. Construire des features synthétiques et un "score qualité" à apprendre
# -------------------------
# Pour chaque recette, nous calculons des features dépendant du marché moyen:
def compute_market_summary_for_recipe(recipe, producers_df):
    # moyenne du prix par ingrédient (sur tous les producteurs)
    ing_stats = {}
    for ing, qty in recipe['ingredients'].items():
        dfp = producers_df[producers_df['ingredient_id']==ing]
        if dfp.shape[0]==0:
            # ingrédient absent -> pénalité
            ing_stats[ing] = {"price_mean": 999, "quality_mean": 0, "availability": 0}
        else:
            ing_stats[ing] = {
                "price_mean": dfp['price'].mean(),
                "quality_mean": dfp['quality'].mean(),
                "availability": dfp['qty'].sum()
            }
    return ing_stats

# créer dataset d'entraînement synthétique: on simule plusieurs "marchés" aléatoires (variantes de producteurs)
train_rows = []
for market_variant in range(200):
    # perturber légèrement les prix/qualités
    dfp = df_producers.copy()
    dfp['price'] = (dfp['price'] * np.random.normal(1.0, 0.15, size=len(dfp))).clip(0.1, None)
    dfp['quality'] = dfp['quality'].apply(lambda x: min(1.0, max(0.2, x + np.random.normal(0,0.08))))
    # for each recipe compute features and a synthetic "quality score"
    for _, rec in df_recipes.iterrows():
        stats = compute_market_summary_for_recipe(rec, dfp)
        total_cost = 0.0
        weighted_quality = 0.0
        availability_flag = 1.0
        for ing, qty in rec['ingredients'].items():
            ps = stats[ing]['price_mean']
            qs = stats[ing]['quality_mean']
            av = stats[ing]['availability']
            total_cost += ps * qty
            weighted_quality += qs * qty
            if av < qty:  # not enough stock
                availability_flag = 0.0
        # synthetic quality score: combines ingredient quality, inverse cost, time factor, availability
        base_quality = (weighted_quality / sum(rec['ingredients'].values())) * availability_flag
        cost_penalty = 1 / (1 + math.log1p(total_cost))
        time_penalty = 1 / (1 + rec['time']/60)  # prefer faster
        # add some noise
        score = (0.6 * base_quality + 0.3 * cost_penalty + 0.1 * time_penalty) + np.random.normal(0,0.02)
        score = float(max(0.0, min(1.0, score)))
        row = {
            "recipe_id": rec['id'],
            "recipe_time": rec['time'],
            "cost_estimate": total_cost,
            "weighted_quality": weighted_quality,
            "availability_flag": availability_flag,
            "score": score
        }
        # add per-ingredient price and quality as features (flatten)
        for ing in df_ingredients['ingredient_id']:
            row[f"price_{ing}"] = stats.get(ing, {}).get('price_mean', 999)
            row[f"quality_{ing}"] = stats.get(ing, {}).get('quality_mean', 0)
            row[f"avail_{ing}"] = stats.get(ing, {}).get('availability', 0)
        train_rows.append(row)

df_train = pd.DataFrame(train_rows)

# -------------------------
# 3. Entraînement d'un modèle simple pour prédire le score
# -------------------------
feature_cols = [c for c in df_train.columns if c not in ['score','recipe_id']]
X = df_train[feature_cols]
y = df_train['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
print("Model trained. Test R2:", model.score(X_test, y_test))

# -------------------------
# 4. Fonction d'inférence & allocation d'achat
# -------------------------
def suggest_recipes_for_user(user_region, user_budget, df_recipes, df_producers_current, model, top_k=3):
    suggestions = []
    # pour chaque recette calculer features selon marché actuel (df_producers_current)
    for _, rec in df_recipes.iterrows():
        stats = compute_market_summary_for_recipe(rec, df_producers_current)
        total_cost = 0.0
        weighted_quality = 0.0
        availability_flag = 1.0
        for ing, qty in rec['ingredients'].items():
            ps = stats[ing]['price_mean']
            qs = stats[ing]['quality_mean']
            av = stats[ing]['availability']
            total_cost += ps * qty
            weighted_quality += qs * qty
            if av < qty:
                availability_flag = 0.0
        # prepare feature vector
        row = {
            "recipe_time": rec['time'],
            "cost_estimate": total_cost,
            "weighted_quality": weighted_quality,
            "availability_flag": availability_flag
        }
        for ing in df_ingredients['ingredient_id']:
            row[f"price_{ing}"] = stats.get(ing, {}).get('price_mean', 999)
            row[f"quality_{ing}"] = stats.get(ing, {}).get('quality_mean', 0)
            row[f"avail_{ing}"] = stats.get(ing, {}).get('availability', 0)
        X_row = pd.DataFrame([row])[feature_cols]
        pred_score = float(model.predict(X_row)[0])
        # check budget constraint: we interpret user_budget as max cost per daily meal
        if total_cost <= user_budget and availability_flag>0:
            suggestions.append({
                "recipe_id": rec['id'],
                "name": rec['name'],
                "predicted_score": pred_score,
                "cost_estimate": total_cost,
                "ingredients": rec['ingredients']
            })
    # sort by predicted_score desc
    suggestions = sorted(suggestions, key=lambda x: x['predicted_score'], reverse=True)
    # For each suggestion, compute optimized allocation among producers (min cost)
    final_suggestions = []
    for s in suggestions[:top_k]:
        allocation, total_cost_alloc = optimize_purchase_allocation(s['ingredients'], df_producers_current, user_region)
        s['allocation'] = allocation
        s['total_cost_alloc'] = total_cost_alloc
        final_suggestions.append(s)
    return final_suggestions

def optimize_purchase_allocation(ingredients_needed, producers_df, user_region):
    # Simple linear program: minimize cost sum(price_i * x_i)
    # subject to: sum x_i >= qty_needed for each ingredient, 0 <= x_i <= producer_qty
    # optional: prefer local producers (same region) by tiny cost bonus
    prob = pulp.LpProblem("min_cost", pulp.LpMinimize)
    # variables x_producer = qty bought from that producer
    var_dict = {}
    for idx, p in producers_df.iterrows():
        pid = p['producer_id']
        var_dict[pid] = pulp.LpVariable(f"x_{pid}", lowBound=0, upBound=p['qty'], cat='Continuous')
    # objective
    cost_terms = []
    for idx, p in producers_df.iterrows():
        pid = p['producer_id']
        price = p['price']
        # small discount for same-region producers -> encourage local purchases:
        local_bonus = -0.01 if p['region'] == user_region else 0.0
        cost_terms.append((price + local_bonus) * var_dict[pid])
    prob += pulp.lpSum(cost_terms)
    # constraints per ingredient
    for ing, qty_needed in ingredients_needed.items():
        prob += pulp.lpSum([var_dict[p['producer_id']] for _, p in producers_df[producers_df['ingredient_id']==ing].iterrows()]) >= qty_needed
    # solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    allocation = {}
    total_cost = 0.0
    for _, p in producers_df.iterrows():
        pid = p['producer_id']
        val = var_dict[pid].varValue if var_dict[pid].varValue is not None else 0.0
        if val > 1e-6:
            allocation[pid] = {"ingredient_id": p['ingredient_id'], "qty": round(val,3), "price": p['price'], "region": p['region']}
            total_cost += val * p['price']
    return allocation, round(total_cost,3)

# -------------------------
# 5. Exemple d'utilisation (marché actuel = df_producers, utilisateur)
# -------------------------
# Simuler marché courant (peut être simplement df_producers)
df_producers_current = df_producers.copy()

# Exemple utilisateur
user_region = "Analamanga"
user_budget = 1.5  # budget par recette (ex: 1.5 unités monétaires)

suggestions = suggest_recipes_for_user(user_region=user_region, user_budget=user_budget,
                                       df_recipes=df_recipes, df_producers_current=df_producers_current,
                                       model=model, top_k=3)

print("\nSuggestions pour l'utilisateur (region={}, budget={}):\n".format(user_region, user_budget))
for s in suggestions:
    print(f"- {s['name']} (score attendu {s['predicted_score']:.3f}) -> coût estimé {s['total_cost_alloc']:.2f}")
    print("  Ingrédients:", s['ingredients'])
    print("  Allocation (producteur: qty @ price):")
    for pid, alloc in s['allocation'].items():
        print(f"    {pid} : {alloc['ingredient_id']} {alloc['qty']} @ {alloc['price']} (region {alloc['region']})")
    print("")

# Le script affiche des suggestions et allocations.

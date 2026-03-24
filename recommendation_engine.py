"""
Advanced Meal Recommendation Engine
Uses: OpenAI Embeddings + Vector Search + Collaborative Filtering + Feedback Loop
"""

import numpy as np
import pandas as pd
import json
import logging
import pickle
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import openai

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. EMBEDDING ENGINE  (semantic meal similarity)
# ─────────────────────────────────────────────
class EmbeddingEngine:
    """
    Converts meals into dense vector representations using OpenAI embeddings,
    enabling semantic similarity search far beyond keyword matching.
    """

    EMBED_MODEL = "text-embedding-3-small"

    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.meal_embeddings: Dict[str, np.ndarray] = {}   # meal_id → vector
        self.meal_metadata: Dict[str, dict] = {}           # meal_id → raw data

    # ── build ────────────────────────────────
    def build_meal_text(self, meal_row: dict) -> str:
        """Rich text representation fed to the embedding model."""
        parts = [
            f"Meal: {meal_row.get('Dish Name', '')}",
            f"Type: {meal_row.get('Meal Time', '')}",
            f"Diet: {meal_row.get('Diet Type', '')}",
            f"Cuisine: {meal_row.get('Cuisine', '')}",
            f"Ingredients: {meal_row.get('Ingredients', '')}",
            f"Calories: {meal_row.get('Calories', '')}",
            f"Protein: {meal_row.get('Protein(g)', '')}g",
            f"Carbs: {meal_row.get('Carbs(g)', '')}g",
            f"Fat: {meal_row.get('Fat(g)', '')}g",
            f"Tags: {meal_row.get('Tags', '')}",
        ]
        return " | ".join(p for p in parts if p.split(": ", 1)[-1])

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text (used for query-time only)."""
        response = self.client.embeddings.create(model=self.EMBED_MODEL, input=text)
        return np.array(response.data[0].embedding)

    # ── cache helpers ────────────────────────
    def save_index(self, path: str = "embeddings_cache.pkl") -> None:
        """Save embeddings + metadata to disk so next startup is instant."""
        with open(path, "wb") as f:
            pickle.dump({
                "embeddings": self.meal_embeddings,
                "metadata":   self.meal_metadata,
            }, f)
        logger.info(f"💾 Embedding cache saved → {path}")

    def load_index(self, path: str = "embeddings_cache.pkl") -> bool:
        """Load cached embeddings if file exists. Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.meal_embeddings = data["embeddings"]
            self.meal_metadata   = data["metadata"]
            logger.info(f"⚡ Loaded {len(self.meal_embeddings)} embeddings from cache (0 API calls)")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), rebuilding …")
            return False

    # ── main build ───────────────────────────
    def build_index(
        self,
        meals_df: pd.DataFrame,
        cache_path: str = "embeddings_cache.pkl",
        batch_size: int = 100,
    ) -> None:
        """
        Build embedding index with:
          • Cache check  → 0 API calls on subsequent startups  (instant ⚡)
          • Batch embed  → ~32 calls instead of 3 114 on first run
        """
        # ── Try cache first ──────────────────
        if self.load_index(cache_path):
            return

        logger.info(f"Building embedding index for {len(meals_df)} meals (batch_size={batch_size}) …")

        # ── Collect texts & metadata ─────────
        meal_ids: List[str] = []
        texts:    List[str] = []

        for _, row in meals_df.iterrows():
            meal_id = str(row.get("Dish Name", "")).strip()
            if not meal_id:
                continue
            meal_ids.append(meal_id)
            texts.append(self.build_meal_text(row.to_dict()))
            self.meal_metadata[meal_id] = row.to_dict()

        # ── Batch embed ──────────────────────
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids   = meal_ids[i : i + batch_size]

            response = self.client.embeddings.create(
                model=self.EMBED_MODEL,
                input=batch_texts,
            )

            for j, emb_data in enumerate(response.data):
                self.meal_embeddings[batch_ids[j]] = np.array(emb_data.embedding)

            done = min(i + batch_size, total)
            logger.info(f"  Embedded {done}/{total} meals …")

        logger.info(f"✅ Embedding index ready — {len(self.meal_embeddings)} meals indexed.")

        # ── Persist for next restart ─────────
        self.save_index(cache_path)

    # ── query ────────────────────────────────
    def find_similar_meals(
        self,
        query: str,
        top_k: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return (meal_id, similarity_score) sorted descending."""
        if not self.meal_embeddings:
            return []

        query_vec = self.embed_text(query).reshape(1, -1)
        exclude = set(exclude_ids or [])

        ids, vecs = [], []
        for mid, vec in self.meal_embeddings.items():
            if mid not in exclude:
                ids.append(mid)
                vecs.append(vec)

        if not vecs:
            return []

        scores = cosine_similarity(query_vec, np.array(vecs))[0]
        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_meal_embedding(self, meal_id: str) -> Optional[np.ndarray]:
        return self.meal_embeddings.get(meal_id)


# ─────────────────────────────────────────────
# 2. USER PROFILE BUILDER  (taste model)
# ─────────────────────────────────────────────
class UserTasteProfileBuilder:
    """
    Maintains a running taste embedding per user by averaging the embeddings
    of meals they liked, weighted by rating and recency.
    """

    DECAY_DAYS = 30  # older interactions contribute less

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.emb = embedding_engine
        # uid → {"taste_vector": np.ndarray, "interactions": [...]}
        self.profiles: Dict[str, dict] = {}

    def _decay_weight(self, timestamp_str: str) -> float:
        try:
            ts = datetime.fromisoformat(timestamp_str)
            days_ago = (datetime.utcnow() - ts).days
            return max(0.1, 1.0 - days_ago / self.DECAY_DAYS)
        except Exception:
            return 1.0

    def update_profile(self, uid: str, interactions: List[dict]) -> None:
        """
        interactions: list of
          {"meal_id": str, "rating": float (1-5), "timestamp": ISO-str}
        """
        positive = [i for i in interactions if i.get("rating", 0) >= 3]
        if not positive:
            return

        weighted_vecs, total_weight = [], 0.0
        for inter in positive:
            vec = self.emb.get_meal_embedding(inter["meal_id"])
            if vec is None:
                continue
            w = (inter["rating"] / 5.0) * self._decay_weight(inter["timestamp"])
            weighted_vecs.append(vec * w)
            total_weight += w

        if not weighted_vecs or total_weight == 0:
            return

        taste_vec = np.sum(weighted_vecs, axis=0) / total_weight
        self.profiles[uid] = {
            "taste_vector": taste_vec,
            "interactions": interactions,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def get_taste_vector(self, uid: str) -> Optional[np.ndarray]:
        p = self.profiles.get(uid)
        return p["taste_vector"] if p else None

    def get_disliked_meals(self, uid: str) -> List[str]:
        p = self.profiles.get(uid, {})
        return [
            i["meal_id"]
            for i in p.get("interactions", [])
            if i.get("rating", 5) <= 2
        ]


# ─────────────────────────────────────────────
# 3. COLLABORATIVE FILTER  (user-user similarity)
# ─────────────────────────────────────────────
class CollaborativeFilter:
    """
    User-user collaborative filtering:
    'People with a similar taste profile also loved these meals.'
    """

    def __init__(self, taste_builder: UserTasteProfileBuilder):
        self.taste = taste_builder

    def find_similar_users(self, uid: str, top_k: int = 5) -> List[str]:
        target_vec = self.taste.get_taste_vector(uid)
        if target_vec is None:
            return []

        similarities = []
        for other_uid, profile in self.taste.profiles.items():
            if other_uid == uid:
                continue
            sim = cosine_similarity(
                target_vec.reshape(1, -1),
                profile["taste_vector"].reshape(1, -1),
            )[0][0]
            similarities.append((other_uid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [u for u, _ in similarities[:top_k]]

    def get_collaborative_meals(
        self,
        uid: str,
        already_seen: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return meals highly rated by similar users, not yet seen by uid."""
        similar_users = self.find_similar_users(uid)
        if not similar_users:
            return []

        seen = set(already_seen or [])
        meal_scores: Dict[str, float] = defaultdict(float)

        for i, other_uid in enumerate(similar_users):
            weight = 1.0 / (i + 1)          # closer users get higher weight
            profile = self.taste.profiles.get(other_uid, {})
            for inter in profile.get("interactions", []):
                if inter["meal_id"] not in seen and inter.get("rating", 0) >= 4:
                    meal_scores[inter["meal_id"]] += inter["rating"] * weight

        ranked = sorted(meal_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ─────────────────────────────────────────────
# 4. NUTRITION SCORER  (health-goal alignment)
# ─────────────────────────────────────────────
class NutritionScorer:
    """
    Scores each meal against the user's calorie budget and macro targets.
    Returns a 0-1 score per meal.
    """

    def score_meal(
        self,
        meal: dict,
        target_calories: float,
        target_protein: float,
        target_carbs: float,
        target_fat: float,
        tolerance: float = 0.20,
    ) -> float:
        scores = []

        def _pct_score(actual, target):
            if target <= 0:
                return 1.0
            diff = abs(actual - target) / target
            return max(0.0, 1.0 - diff / tolerance)

        try:
            scores.append(_pct_score(float(meal.get("Calories", 0)), target_calories))
            scores.append(_pct_score(float(meal.get("Protein(g)", 0)), target_protein))
            scores.append(_pct_score(float(meal.get("Carbs(g)", 0)), target_carbs))
            scores.append(_pct_score(float(meal.get("Fat(g)", 0)), target_fat))
        except (ValueError, TypeError):
            return 0.5

        return float(np.mean(scores))

    def compute_daily_targets(self, calorie_range: str, health_goals: List[str]) -> dict:
        """Parse calorie range and split into per-meal macro targets."""
        try:
            low, high = map(int, calorie_range.split("-"))
            daily_calories = (low + high) / 2
        except Exception:
            daily_calories = 2000

        # Adjust macros based on health goals
        if any("muscle" in g.lower() or "gain" in g.lower() for g in health_goals):
            protein_pct, carb_pct, fat_pct = 0.35, 0.40, 0.25
        elif any("weight loss" in g.lower() or "lose" in g.lower() for g in health_goals):
            protein_pct, carb_pct, fat_pct = 0.35, 0.35, 0.30
        elif any("keto" in g.lower() or "low carb" in g.lower() for g in health_goals):
            protein_pct, carb_pct, fat_pct = 0.30, 0.10, 0.60
        else:
            protein_pct, carb_pct, fat_pct = 0.30, 0.40, 0.30

        # Distribute across 5 meals (not evenly — bigger for lunch/dinner)
        meal_splits = {
            "Breakfast": 0.20,
            "Mid-Morning Snack": 0.10,
            "Lunch": 0.30,
            "Afternoon Snack": 0.10,
            "Dinner": 0.30,
        }

        targets = {}
        for meal_type, split in meal_splits.items():
            cal = daily_calories * split
            targets[meal_type] = {
                "calories": cal,
                "protein": (cal * protein_pct) / 4,
                "carbs": (cal * carb_pct) / 4,
                "fat": (cal * fat_pct) / 9,
            }
        return targets


# ─────────────────────────────────────────────
# 5. HYBRID RANKER  (combines all signals)
# ─────────────────────────────────────────────
class HybridRanker:
    """
    Final scoring = weighted sum of:
      • Content-based (embedding similarity to user taste)
      • Collaborative (liked by similar users)
      • Nutrition fit (macro/calorie alignment)
      • Popularity (global rating)
      • Novelty (penalise recently shown meals)
    """

    WEIGHTS = {
        "content":       0.35,
        "collaborative": 0.25,
        "nutrition":     0.25,
        "popularity":    0.10,
        "novelty":       0.05,
    }

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        taste_builder: UserTasteProfileBuilder,
        collab_filter: CollaborativeFilter,
        nutrition_scorer: NutritionScorer,
    ):
        self.emb = embedding_engine
        self.taste = taste_builder
        self.collab = collab_filter
        self.nutr = nutrition_scorer

    def rank_meals(
        self,
        uid: str,
        candidate_meals: List[dict],
        meal_type: str,
        calorie_range: str,
        health_goals: List[str],
        recently_shown: Optional[List[str]] = None,
    ) -> List[dict]:
        if not candidate_meals:
            return []

        targets = self.nutr.compute_daily_targets(calorie_range, health_goals)
        meal_target = targets.get(meal_type, targets.get("Lunch", {}))

        # Pre-compute collaborative signal
        collab_meals = dict(self.collab.get_collaborative_meals(uid))
        collab_max = max(collab_meals.values(), default=1)

        # Taste vector
        taste_vec = self.taste.get_taste_vector(uid)

        recently_shown_set = set(recently_shown or [])

        scored = []
        for meal in candidate_meals:
            meal_id = meal.get("Dish Name", "")
            s = {}

            # Content score
            if taste_vec is not None:
                meal_vec = self.emb.get_meal_embedding(meal_id)
                if meal_vec is not None:
                    s["content"] = float(
                        cosine_similarity(taste_vec.reshape(1, -1), meal_vec.reshape(1, -1))[0][0]
                    )
                else:
                    s["content"] = 0.5
            else:
                s["content"] = 0.5

            # Collaborative score
            raw_collab = collab_meals.get(meal_id, 0)
            s["collaborative"] = raw_collab / collab_max if collab_max else 0

            # Nutrition score
            s["nutrition"] = self.nutr.score_meal(
                meal,
                meal_target.get("calories", 400),
                meal_target.get("protein", 30),
                meal_target.get("carbs", 50),
                meal_target.get("fat", 15),
            )

            # Popularity score (use avg_rating col if available)
            try:
                s["popularity"] = float(meal.get("Rating", 3)) / 5.0
            except Exception:
                s["popularity"] = 0.6

            # Novelty (penalise recently shown)
            s["novelty"] = 0.0 if meal_id in recently_shown_set else 1.0

            # Weighted final score
            final = sum(self.WEIGHTS[k] * v for k, v in s.items())
            scored.append({**meal, "_score": round(final, 4), "_signals": s})

        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored


# ─────────────────────────────────────────────
# 6. FEEDBACK MANAGER
# ─────────────────────────────────────────────
class FeedbackManager:
    """
    Persists user interactions (ratings, skips, repeats) and triggers
    profile re-computation after each batch of feedback.
    """

    def __init__(self, taste_builder: UserTasteProfileBuilder):
        self.taste = taste_builder
        # uid → list of interaction dicts
        self._store: Dict[str, List[dict]] = defaultdict(list)

    def record(
        self,
        uid: str,
        meal_id: str,
        rating: float,           # 1-5
        action: str = "rated",   # "rated" | "skipped" | "repeated"
    ) -> None:
        entry = {
            "meal_id": meal_id,
            "rating": rating,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._store[uid].append(entry)
        # Rebuild taste profile immediately
        self.taste.update_profile(uid, self._store[uid])
        logger.info(f"Feedback recorded for {uid}: {meal_id} → {rating}★")

    def get_history(self, uid: str) -> List[dict]:
        return self._store.get(uid, [])

    def get_seen_meals(self, uid: str, days: int = 7) -> List[str]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [
            i["meal_id"]
            for i in self._store.get(uid, [])
            if datetime.fromisoformat(i["timestamp"]) > cutoff
        ]

    def export_for_firebase(self, uid: str) -> dict:
        return {
            "uid": uid,
            "interactions": self._store.get(uid, []),
            "total_ratings": len(self._store.get(uid, [])),
        }


# ─────────────────────────────────────────────
# 7. ADVANCED RECOMMENDATION ORCHESTRATOR
# ─────────────────────────────────────────────
class AdvancedMealRecommender:
    """
    Top-level orchestrator.  Drop-in replacement for the simple
    filter_meals_by_preferences + LLM approach.

    Usage:
        recommender = AdvancedMealRecommender(openai_client)
        recommender.initialise(meals_df)

        # record user feedback
        recommender.record_feedback(uid, meal_id, rating=4)

        # get ranked meals for a meal slot
        top_meals = recommender.recommend(uid, user_profile, meal_type="Lunch")
    """

    def __init__(self, client: openai.OpenAI):
        self.emb   = EmbeddingEngine(client)
        self.taste = UserTasteProfileBuilder(self.emb)
        self.collab = CollaborativeFilter(self.taste)
        self.nutr  = NutritionScorer()
        self.ranker = HybridRanker(self.emb, self.taste, self.collab, self.nutr)
        self.feedback = FeedbackManager(self.taste)
        self._meals_df: Optional[pd.DataFrame] = None

    # ── startup ──────────────────────────────
    def initialise(self, meals_df: pd.DataFrame) -> None:
        self._meals_df = meals_df
        self.emb.build_index(meals_df)

    # ── feedback ─────────────────────────────
    def record_feedback(
        self, uid: str, meal_id: str, rating: float, action: str = "rated"
    ) -> dict:
        self.feedback.record(uid, meal_id, rating, action)
        return {"status": "ok", "meal_id": meal_id, "rating": rating}

    # ── recommend ────────────────────────────
    def recommend(
        self,
        uid: str,
        user_profile,            # your existing UserProfile pydantic model
        meal_type: str,          # "Breakfast" | "Lunch" | …
        calorie_range: str,
        top_k: int = 5,
        candidate_pool: int = 40,
    ) -> List[dict]:
        if self._meals_df is None or self._meals_df.empty:
            return []

        disliked = self.taste.get_disliked_meals(uid)
        recently_shown = self.feedback.get_seen_meals(uid, days=7)

        # ── Step 1: semantic candidates ──────
        query = self._build_query(user_profile, meal_type)
        similar = self.emb.find_similar_meals(
            query,
            top_k=candidate_pool,
            exclude_ids=disliked,
        )
        candidate_ids = {mid for mid, _ in similar}

        # ── Step 2: add collaborative candidates ──
        collab_meals = self.collab.get_collaborative_meals(uid, disliked, top_k=20)
        candidate_ids.update(mid for mid, _ in collab_meals)

        # ── Step 3: filter & build candidate list ──
        candidates = [
            self.emb.meal_metadata[mid]
            for mid in candidate_ids
            if mid in self.emb.meal_metadata
            and mid not in disliked
        ]

        # ── Step 4: hybrid rank ──────────────
        ranked = self.ranker.rank_meals(
            uid=uid,
            candidate_meals=candidates,
            meal_type=meal_type,
            calorie_range=calorie_range,
            health_goals=user_profile.health_goals,
            recently_shown=recently_shown,
        )
        return ranked[:top_k]

    def recommend_full_day(
        self,
        uid: str,
        user_profile,
        calorie_range: str,
    ) -> Dict[str, List[dict]]:
        """Return top recommendations for all 5 meal slots."""
        meal_types = [
            "Breakfast",
            "Mid-Morning Snack",
            "Lunch",
            "Afternoon Snack",
            "Dinner",
        ]
        return {
            mt: self.recommend(uid, user_profile, mt, calorie_range)
            for mt in meal_types
        }

    # ── helpers ──────────────────────────────
    def _build_query(self, user_profile, meal_type: str) -> str:
        goals = ", ".join(user_profile.health_goals) if user_profile.health_goals else "general wellness"
        allergies = ", ".join(user_profile.food_allergies) if user_profile.food_allergies else "none"
        return (
            f"{meal_type} meal for a {user_profile.diet.value} person "
            f"aged {user_profile.age}, activity: {user_profile.activity_level.value}, "
            f"health goals: {goals}, allergies: {allergies}"
        )

    def get_user_insights(self, uid: str) -> dict:
        """Return analytics about a user's taste profile."""
        history = self.feedback.get_history(uid)
        if not history:
            return {"message": "No feedback recorded yet."}

        ratings = [i["rating"] for i in history]
        liked = [i["meal_id"] for i in history if i["rating"] >= 4]
        disliked = [i["meal_id"] for i in history if i["rating"] <= 2]

        return {
            "total_interactions": len(history),
            "average_rating": round(np.mean(ratings), 2),
            "liked_meals": liked[-10:],
            "disliked_meals": disliked[-10:],
            "has_taste_profile": self.taste.get_taste_vector(uid) is not None,
        }
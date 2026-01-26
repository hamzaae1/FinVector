"""
FinVector - Feedback Loop & Continuous Learning Module
Tracks user interactions and builds preference profiles for personalized recommendations
"""
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading

# Storage directory
FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), 'feedback_data')
os.makedirs(FEEDBACK_DIR, exist_ok=True)


class UserInteraction:
    """Represents a single user interaction"""
    def __init__(self, user_id: str, product_id: str, action: str,
                 query: str = "", metadata: Dict = None):
        self.user_id = user_id
        self.product_id = product_id
        self.action = action  # click, view, add_to_cart, purchase, ignore
        self.query = query
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()


class FeedbackStore:
    """Persistent storage for user interactions and preferences"""

    def __init__(self):
        self.interactions_file = os.path.join(FEEDBACK_DIR, 'interactions.jsonl')
        self.preferences_file = os.path.join(FEEDBACK_DIR, 'preferences.json')
        self.lock = threading.RLock()
        self._load_preferences()

    def _load_preferences(self):
        """Load user preferences from disk"""
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, 'r') as f:
                    self.preferences = json.load(f)
            else:
                self.preferences = {}
        except:
            self.preferences = {}

    def _save_preferences(self):
        """Save preferences to disk"""
        with open(self.preferences_file, 'w') as f:
            json.dump(self.preferences, f, indent=2)

    def record_interaction(self, interaction: UserInteraction):
        """Record a user interaction"""
        with self.lock:
            # Append to interactions log
            with open(self.interactions_file, 'a') as f:
                f.write(json.dumps({
                    'user_id': interaction.user_id,
                    'product_id': interaction.product_id,
                    'action': interaction.action,
                    'query': interaction.query,
                    'metadata': interaction.metadata,
                    'timestamp': interaction.timestamp
                }) + '\n')

            # Update preferences
            self._update_preferences(interaction)

    def _update_preferences(self, interaction: UserInteraction):
        """Update user preferences based on interaction"""
        user_id = interaction.user_id

        if user_id not in self.preferences:
            self.preferences[user_id] = {
                'categories': defaultdict(float),
                'price_range': {'min': float('inf'), 'max': 0, 'avg_viewed': []},
                'brands': defaultdict(float),
                'interaction_count': 0,
                'last_active': None
            }

        prefs = self.preferences[user_id]
        prefs['interaction_count'] += 1
        prefs['last_active'] = interaction.timestamp

        # Weight by action type
        action_weights = {
            'purchase': 5.0,
            'add_to_cart': 3.0,
            'click': 1.0,
            'view': 0.5,
            'ignore': -0.5
        }
        weight = action_weights.get(interaction.action, 0.5)

        # Update category preferences
        category = interaction.metadata.get('category', '')
        if category:
            if isinstance(prefs['categories'], dict):
                prefs['categories'][category] = prefs['categories'].get(category, 0) + weight

        # Update price preferences
        price = interaction.metadata.get('price', 0)
        if price > 0:
            if isinstance(prefs['price_range'], dict):
                prefs['price_range']['min'] = min(prefs['price_range'].get('min', float('inf')), price)
                prefs['price_range']['max'] = max(prefs['price_range'].get('max', 0), price)
                avg_list = prefs['price_range'].get('avg_viewed', [])
                avg_list.append(price)
                # Keep last 50 prices
                prefs['price_range']['avg_viewed'] = avg_list[-50:]

        self._save_preferences()

    def get_user_preferences(self, user_id: str) -> Dict:
        """Get preferences for a user"""
        with self.lock:
            return self.preferences.get(user_id, {})

    def get_recent_interactions(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get recent interactions for a user"""
        interactions = []
        try:
            with open(self.interactions_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data['user_id'] == user_id:
                            interactions.append(data)
                    except:
                        continue
        except FileNotFoundError:
            pass

        return interactions[-limit:]


class PreferenceEngine:
    """Engine for computing personalized recommendations based on preferences"""

    def __init__(self, store: FeedbackStore):
        self.store = store

    def get_category_boost(self, user_id: str) -> Dict[str, float]:
        """Get category preference boosts for a user"""
        prefs = self.store.get_user_preferences(user_id)
        categories = prefs.get('categories', {})

        if not categories:
            return {}

        # Normalize to boost factors (0.8 to 1.5)
        max_score = max(categories.values()) if categories else 1
        return {
            cat: 0.8 + (0.7 * score / max_score)
            for cat, score in categories.items()
        }

    def get_preferred_price_range(self, user_id: str) -> tuple:
        """Get preferred price range based on history"""
        prefs = self.store.get_user_preferences(user_id)
        price_range = prefs.get('price_range', {})

        avg_prices = price_range.get('avg_viewed', [])
        if not avg_prices:
            return (0, float('inf'))

        avg = sum(avg_prices) / len(avg_prices)
        # Preferred range: 50% to 150% of average viewed price
        return (avg * 0.5, avg * 1.5)

    def compute_personalization_score(self, user_id: str, product: Dict) -> float:
        """Compute personalization score for a product"""
        if not user_id:
            return 1.0

        prefs = self.store.get_user_preferences(user_id)
        if not prefs:
            return 1.0

        score = 1.0

        # Category boost
        categories = prefs.get('categories', {})
        product_category = product.get('category', '')
        if product_category in categories:
            max_cat = max(categories.values()) if categories else 1
            score *= 1 + (0.3 * categories[product_category] / max_cat)

        # Price preference
        price_range = prefs.get('price_range', {})
        avg_prices = price_range.get('avg_viewed', [])
        if avg_prices:
            avg = sum(avg_prices) / len(avg_prices)
            product_price = product.get('price', 0)
            if product_price > 0:
                # Products closer to average get higher scores
                price_diff = abs(product_price - avg) / avg if avg > 0 else 0
                score *= max(0.7, 1 - (price_diff * 0.3))

        return min(1.5, score)  # Cap at 1.5x boost


class FeedbackLoop:
    """Main feedback loop system"""

    def __init__(self):
        self.store = FeedbackStore()
        self.engine = PreferenceEngine(self.store)

    def record_click(self, user_id: str, product_id: str, product_data: Dict, query: str = ""):
        """Record a product click"""
        interaction = UserInteraction(
            user_id=user_id,
            product_id=product_id,
            action='click',
            query=query,
            metadata={
                'category': product_data.get('category', ''),
                'price': product_data.get('price', 0),
                'title': product_data.get('title', '')
            }
        )
        self.store.record_interaction(interaction)

    def record_add_to_cart(self, user_id: str, product_id: str, product_data: Dict):
        """Record add to cart action"""
        interaction = UserInteraction(
            user_id=user_id,
            product_id=product_id,
            action='add_to_cart',
            metadata={
                'category': product_data.get('category', ''),
                'price': product_data.get('price', 0),
                'title': product_data.get('title', '')
            }
        )
        self.store.record_interaction(interaction)

    def record_purchase(self, user_id: str, product_id: str, product_data: Dict):
        """Record a purchase"""
        interaction = UserInteraction(
            user_id=user_id,
            product_id=product_id,
            action='purchase',
            metadata={
                'category': product_data.get('category', ''),
                'price': product_data.get('price', 0),
                'title': product_data.get('title', '')
            }
        )
        self.store.record_interaction(interaction)

    def apply_personalization(self, user_id: str, results: List[Dict]) -> List[Dict]:
        """Apply personalization to search results"""
        if not user_id or not results:
            return results

        # Compute personalization scores
        for product in results:
            boost = self.engine.compute_personalization_score(user_id, product)
            product['personalization_boost'] = round(boost, 3)
            # Adjust similarity score with personalization
            original_score = product.get('similarity_score', 0)
            product['personalized_score'] = round(original_score * boost, 3)

        # Re-sort by personalized score
        results.sort(key=lambda x: x.get('personalized_score', 0), reverse=True)

        return results

    def get_user_profile(self, user_id: str) -> Dict:
        """Get user profile summary"""
        prefs = self.store.get_user_preferences(user_id)
        recent = self.store.get_recent_interactions(user_id, limit=10)

        # Get top categories
        categories = prefs.get('categories', {})
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]

        # Get price stats
        price_range = prefs.get('price_range', {})
        avg_prices = price_range.get('avg_viewed', [])
        avg_price = sum(avg_prices) / len(avg_prices) if avg_prices else 0

        return {
            'user_id': user_id,
            'interaction_count': prefs.get('interaction_count', 0),
            'last_active': prefs.get('last_active'),
            'top_categories': [{'category': c, 'score': round(s, 2)} for c, s in top_categories],
            'avg_price_viewed': round(avg_price, 2),
            'price_range': {
                'min': round(price_range.get('min', 0), 2) if price_range.get('min', float('inf')) != float('inf') else 0,
                'max': round(price_range.get('max', 0), 2)
            },
            'recent_actions': len(recent)
        }


# Global instance
feedback_loop = FeedbackLoop()

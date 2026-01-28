"""
FinVector - Query Understanding Module
Intelligently parses and enhances user queries for better search results
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParsedQuery:
    """Structured representation of a parsed user query"""
    original_query: str
    enhanced_query: str
    extracted_budget: Optional[float]
    min_rating: float
    intent: str  # 'budget', 'quality', 'balanced', 'specific'
    categories_hint: List[str]
    attributes: Dict[str, str]
    confidence: float


class QueryUnderstanding:
    """Intelligent query parser and enhancer"""

    # Budget patterns
    BUDGET_PATTERNS = [
        r'under\s*\$?(\d+(?:\.\d{2})?)',
        r'below\s*\$?(\d+(?:\.\d{2})?)',
        r'less than\s*\$?(\d+(?:\.\d{2})?)',
        r'up to\s*\$?(\d+(?:\.\d{2})?)',
        r'max(?:imum)?\s*\$?(\d+(?:\.\d{2})?)',
        r'budget\s*(?:of|is)?\s*\$?(\d+(?:\.\d{2})?)',
        r'\$(\d+(?:\.\d{2})?)\s*(?:or less|max|budget)',
        r'around\s*\$?(\d+(?:\.\d{2})?)',
        r'about\s*\$?(\d+(?:\.\d{2})?)',
        r'(\d+(?:\.\d{2})?)\s*dollars?',
    ]

    # Intent keywords
    BUDGET_INTENT_WORDS = [
        'cheap', 'affordable', 'budget', 'inexpensive', 'economical',
        'low cost', 'value', 'deal', 'bargain', 'save money', 'frugal'
    ]

    QUALITY_INTENT_WORDS = [
        'best', 'premium', 'high quality', 'top rated', 'reliable',
        'durable', 'professional', 'excellent', 'superior', 'luxury'
    ]

    # Synonym expansions for common terms
    SYNONYMS = {
        'wireless': ['bluetooth', 'cordless', 'wifi'],
        'headphones': ['earbuds', 'earphones', 'headset', 'audio'],
        'laptop': ['notebook', 'computer', 'macbook', 'chromebook'],
        'phone': ['smartphone', 'mobile', 'cellphone', 'iphone', 'android'],
        'shoes': ['sneakers', 'footwear', 'boots', 'trainers'],
        'watch': ['timepiece', 'smartwatch', 'wristwatch'],
        'bag': ['backpack', 'purse', 'handbag', 'tote', 'satchel'],
        'jacket': ['coat', 'blazer', 'hoodie', 'sweater', 'outerwear'],
        'pants': ['jeans', 'trousers', 'slacks', 'chinos'],
        'camera': ['dslr', 'mirrorless', 'webcam', 'camcorder'],
        'tv': ['television', 'monitor', 'display', 'screen'],
        'speaker': ['soundbar', 'audio', 'bluetooth speaker'],
        'gaming': ['gamer', 'esports', 'video game', 'playstation', 'xbox'],
        'running': ['jogging', 'marathon', 'athletic', 'sports'],
        'travel': ['luggage', 'suitcase', 'vacation', 'trip'],
        'work': ['office', 'business', 'professional', 'workplace'],
        'kids': ['children', 'toddler', 'baby', 'youth', 'junior'],
    }

    # Context expansions - what to add based on use case
    CONTEXT_EXPANSIONS = {
        'for programming': ['high performance', 'fast processor', 'good ram', 'developer'],
        'for gaming': ['high fps', 'graphics', 'rgb', 'performance', 'gamer'],
        'for work': ['professional', 'business', 'office', 'productivity'],
        'for travel': ['portable', 'lightweight', 'compact', 'carry on'],
        'for running': ['lightweight', 'breathable', 'athletic', 'cushioned'],
        'for gym': ['workout', 'fitness', 'athletic', 'training'],
        'for school': ['student', 'study', 'education', 'college'],
        'for kids': ['children', 'safe', 'durable', 'youth'],
        'for gift': ['popular', 'highly rated', 'bestseller'],
        'for outdoor': ['waterproof', 'durable', 'weather resistant'],
        'for home': ['household', 'domestic', 'indoor'],
    }

    # Category hints based on keywords
    CATEGORY_HINTS = {
        'shoe': ["Men's Shoes"],
        'sneaker': ["Men's Shoes"],
        'boot': ["Men's Shoes"],
        'jacket': ["Men's Clothing"],
        'shirt': ["Men's Clothing"],
        'pants': ["Men's Clothing"],
        'jeans': ["Men's Clothing"],
        'watch': ["Men's Accessories"],
        'belt': ["Men's Accessories"],
        'wallet': ["Men's Accessories"],
        'sunglasses': ["Men's Accessories"],
        'xbox': ["Xbox 360 Games, Consoles & Accessories"],
        'game': ["Xbox 360 Games, Consoles & Accessories"],
        'controller': ["Xbox 360 Games, Consoles & Accessories"],
        'vacuum': ["Vacuum Cleaners & Floor Care"],
        'cleaner': ["Vacuum Cleaners & Floor Care"],
        'luggage': ["Suitcases"],
        'suitcase': ["Suitcases"],
        'travel bag': ["Suitcases"],
    }

    # Attribute patterns
    COLOR_WORDS = ['black', 'white', 'red', 'blue', 'green', 'brown', 'gray', 'grey',
                   'pink', 'purple', 'orange', 'yellow', 'navy', 'beige', 'tan']

    SIZE_WORDS = ['small', 'medium', 'large', 'xl', 'xxl', 'xs', 'mini', 'compact', 'big']

    MATERIAL_WORDS = ['leather', 'cotton', 'wool', 'synthetic', 'mesh', 'canvas',
                      'suede', 'rubber', 'metal', 'plastic', 'wood', 'fabric']

    def parse(self, query: str, default_budget: float = 200.0) -> ParsedQuery:
        """
        Parse a natural language query into structured search parameters

        Args:
            query: User's natural language query
            default_budget: Default budget if not specified in query

        Returns:
            ParsedQuery with extracted parameters and enhanced query
        """
        original = query
        query_lower = query.lower().strip()

        # Extract budget from query
        extracted_budget = self._extract_budget(query_lower)

        # Detect intent
        intent, min_rating = self._detect_intent(query_lower)

        # Extract attributes (color, size, material)
        attributes = self._extract_attributes(query_lower)

        # Get category hints
        categories = self._get_category_hints(query_lower)

        # Clean query (remove budget phrases, intent words)
        clean_query = self._clean_query(query_lower)

        # Expand query with synonyms and context
        enhanced_query = self._expand_query(clean_query)

        # Calculate confidence
        confidence = self._calculate_confidence(
            extracted_budget, intent, categories, attributes
        )

        return ParsedQuery(
            original_query=original,
            enhanced_query=enhanced_query,
            extracted_budget=extracted_budget,
            min_rating=min_rating,
            intent=intent,
            categories_hint=categories,
            attributes=attributes,
            confidence=confidence
        )

    def _extract_budget(self, query: str) -> Optional[float]:
        """Extract budget amount from query"""
        for pattern in self.BUDGET_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    def _detect_intent(self, query: str) -> Tuple[str, float]:
        """Detect user intent and return appropriate min_rating"""
        budget_score = sum(1 for word in self.BUDGET_INTENT_WORDS if word in query)
        quality_score = sum(1 for word in self.QUALITY_INTENT_WORDS if word in query)

        # Check for price mentions (suggests budget-conscious)
        has_price = any(re.search(p, query) for p in self.BUDGET_PATTERNS)

        # Check for gift (suggests quality matters)
        is_gift = 'gift' in query or 'present' in query

        # Check for work/professional context (quality matters)
        is_professional = any(w in query for w in ['work', 'office', 'business', 'professional'])

        if quality_score > budget_score:
            return 'quality', 4.0  # High quality = require good ratings
        elif budget_score > quality_score:
            return 'budget', 3.0  # Budget focus = still want decent quality
        elif budget_score > 0 and quality_score > 0:
            return 'balanced', 3.5  # Want both = moderate rating
        elif has_price:
            return 'value', 3.5  # Price mentioned = want good value
        elif is_gift:
            return 'gift', 4.0  # Gift = should be good quality
        elif is_professional:
            return 'professional', 4.0  # Work = quality matters
        else:
            return 'discovery', 3.5  # General browsing = show decent stuff

    def _extract_attributes(self, query: str) -> Dict[str, str]:
        """Extract product attributes from query"""
        attributes = {}

        # Extract color
        for color in self.COLOR_WORDS:
            if color in query:
                attributes['color'] = color
                break

        # Extract size
        for size in self.SIZE_WORDS:
            if re.search(rf'\b{size}\b', query):
                attributes['size'] = size
                break

        # Extract material
        for material in self.MATERIAL_WORDS:
            if material in query:
                attributes['material'] = material
                break

        return attributes

    def _get_category_hints(self, query: str) -> List[str]:
        """Get category hints based on keywords"""
        categories = []
        for keyword, cats in self.CATEGORY_HINTS.items():
            if keyword in query:
                categories.extend(cats)
        return list(set(categories))

    def _clean_query(self, query: str) -> str:
        """Remove budget phrases and filler words from query"""
        clean = query

        # Remove budget phrases
        for pattern in self.BUDGET_PATTERNS:
            clean = re.sub(pattern, '', clean, flags=re.IGNORECASE)

        # Remove common filler phrases
        filler_phrases = [
            r'\bi need\b', r'\bi want\b', r'\bi\'m looking for\b',
            r'\blooking for\b', r'\bfind me\b', r'\bshow me\b',
            r'\bcan you find\b', r'\bsearch for\b', r'\bget me\b',
            r'\bplease\b', r'\bthanks\b', r'\bthank you\b',
        ]
        for phrase in filler_phrases:
            clean = re.sub(phrase, '', clean, flags=re.IGNORECASE)

        # Remove extra whitespace
        clean = ' '.join(clean.split())

        return clean.strip()

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and context"""
        expanded_terms = [query]

        # Add synonyms
        for word, synonyms in self.SYNONYMS.items():
            if word in query:
                # Add top 2 synonyms
                expanded_terms.extend(synonyms[:2])

        # Add context expansions
        for context, expansions in self.CONTEXT_EXPANSIONS.items():
            if context in query:
                expanded_terms.extend(expansions[:3])

        # Combine unique terms
        all_terms = []
        seen = set()
        for term in expanded_terms:
            for word in term.split():
                if word not in seen and len(word) > 2:
                    seen.add(word)
                    all_terms.append(word)

        return ' '.join(all_terms)

    def _calculate_confidence(
        self,
        budget: Optional[float],
        intent: str,
        categories: List[str],
        attributes: Dict
    ) -> float:
        """Calculate confidence score for the parsing"""
        score = 0.5  # Base confidence

        if budget is not None:
            score += 0.15
        if intent != 'specific':
            score += 0.1
        if categories:
            score += 0.15
        if attributes:
            score += 0.1

        return min(1.0, score)


# Global instance
query_parser = QueryUnderstanding()


def understand_query(query: str, default_budget: float = 200.0) -> ParsedQuery:
    """
    Main function to understand a user query

    Example usage:
        result = understand_query("I need wireless headphones under $50")
        # result.extracted_budget = 50.0
        # result.enhanced_query = "wireless headphones bluetooth earbuds cordless"
        # result.intent = "specific"
    """
    return query_parser.parse(query, default_budget)


# Quick test
if __name__ == "__main__":
    test_queries = [
        "cheap running shoes under $80",
        "best laptop for programming",
        "I need a gift under $50",
        "wireless headphones good quality",
        "black leather jacket for work",
        "affordable travel suitcase",
        "gaming headset reliable",
    ]

    print("=" * 60)
    print("Query Understanding Test")
    print("=" * 60)

    for q in test_queries:
        result = understand_query(q)
        print(f"\nQuery: '{q}'")
        print(f"  Enhanced: '{result.enhanced_query}'")
        print(f"  Budget: ${result.extracted_budget}" if result.extracted_budget else "  Budget: Not specified")
        print(f"  Intent: {result.intent} (min_rating: {result.min_rating})")
        print(f"  Categories: {result.categories_hint or 'None'}")
        print(f"  Attributes: {result.attributes or 'None'}")
        print(f"  Confidence: {result.confidence:.0%}")

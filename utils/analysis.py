# utils/analysis.py

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Initialize VADER once
vader = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment using VADER"""
    scores = vader.polarity_scores(text)
    return {
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu']
    }

def extract_linguistic_features(text: str) -> dict:
    """Extract mental health relevant linguistic features"""
    text_lower = text.lower()
    
    # First-person pronouns
    first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
    pronoun_count = sum(
        text_lower.count(f' {p} ') + 
        text_lower.count(f' {p},') + 
        text_lower.count(f' {p}.') +
        (1 if text_lower.startswith(f'{p} ') else 0)
        for p in first_person_pronouns
    )
    
    # Negative words (depression indicators)
    negative_words = [
        'sad', 'depressed', 'hopeless', 'worthless', 'tired', 'exhausted',
        'alone', 'lonely', 'empty', 'numb', 'helpless', 'useless',
        'horrible', 'terrible', 'awful', 'miserable', 'hate', 'hurt'
    ]
    negative_word_count = sum(text_lower.count(word) for word in negative_words)
    
    # Absolute language
    absolute_words = ['always', 'never', 'nothing', 'nobody', 'none', 'everyone', 'everything']
    absolute_word_count = sum(text_lower.count(f' {word} ') for word in absolute_words)
    
    # Word statistics
    words = re.findall(r'\b\w+\b', text_lower)
    unique_words = set(words)
    lexical_diversity = len(unique_words) / len(words) if len(words) > 0 else 0
    
    # Sentence length
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    return {
        'first_person_pronouns': pronoun_count,
        'negative_words': negative_word_count,
        'absolute_words': absolute_word_count,
        'lexical_diversity': round(lexical_diversity, 3),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'total_words': len(words)
    }

def calculate_wellness_score(emotion: str, sentiment: dict, features: dict) -> float:
    """Calculate wellness score 0-10"""
    emotion_scores = {
        'joy': 8.0,
        'neutral': 5.0,
        'sadness': 3.0,
        'anxiety': 3.5,
        'anger': 2.5
    }
    
    base_score = emotion_scores.get(emotion, 5.0)
    sentiment_adjustment = sentiment['compound'] * 2
    negative_penalty = min(features['negative_words'] * 0.5, 2.0)
    
    wellness_score = base_score + sentiment_adjustment - negative_penalty
    wellness_score = max(0, min(10, wellness_score))
    
    return round(wellness_score, 1)

def get_interpretation(score: float, emotion: str) -> str:
    """Generate human-readable interpretation"""
    if score >= 7.5:
        return f"You seem to be in a positive state with {emotion} emotion. Keep it up!"
    elif score >= 5.0:
        return f"Your emotional state appears balanced, though showing {emotion}."
    elif score >= 3.0:
        return f"You seem to be experiencing {emotion}. Consider talking to someone."
    else:
        return f"Your indicators suggest significant {emotion}. Please reach out for support."
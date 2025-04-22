class College:
    """College model to store college information."""
    def __init__(self, name, abbreviation, avg_rating, reviews, review_count):
        self.name = name
        self.abbreviation = abbreviation
        self.avg_rating = avg_rating
        self.reviews = reviews
        self.review_count = review_count

    def to_dict(self):
        """Convert college object to dictionary."""
        return {
            'name': self.name,
            'abbreviation': self.abbreviation,
            'avg_rating': self.avg_rating,
            'reviews': self.reviews,
            'review_count': self.review_count
        }

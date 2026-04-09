"""
Monitoring des collisions de hash pour optimiser n_features.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Set, List


@dataclass
class CollisionStats:
    """Statistiques de collision"""
    total_tokens: int
    unique_tokens: int
    collision_count: int
    collision_rate: float
    used_buckets: int
    fill_rate: float
    max_bucket_size: int
    recommended_n_features: int


class CollisionTracker:
    """
    Monitore les collisions de hashing.
    Utile pour optimiser INPUT_SIZE.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.bucket_tokens: Dict[int, Set[str]] = defaultdict(set)
        self.collision_count = 0
        self.total_insertions = 0

    def track(self, token: str, index: int) -> bool:
        """
        Enregistre un token et son index.

        Returns:
            True si collision détectée
        """
        self.total_insertions += 1

        is_collision = False
        if index in self.bucket_tokens:
            if token not in self.bucket_tokens[index]:
                # Nouveau token dans bucket existant = collision
                self.collision_count += 1
                is_collision = True

        self.bucket_tokens[index].add(token)
        return is_collision

    def reset(self):
        """Remet à zéro les compteurs"""
        self.bucket_tokens.clear()
        self.collision_count = 0
        self.total_insertions = 0

    def get_stats(self) -> CollisionStats:
        """Calcule les statistiques de collision"""
        used_buckets = len(self.bucket_tokens)
        unique_tokens = sum(len(tokens) for tokens in self.bucket_tokens.values())

        bucket_sizes = [len(tokens) for tokens in self.bucket_tokens.values()]
        max_bucket = max(bucket_sizes) if bucket_sizes else 0

        collision_rate = self.collision_count / max(1, self.total_insertions)
        fill_rate = used_buckets / self.n_features

        # Recommandation
        recommended = self._recommend_size(collision_rate, fill_rate)

        return CollisionStats(
            total_tokens=self.total_insertions,
            unique_tokens=unique_tokens,
            collision_count=self.collision_count,
            collision_rate=collision_rate,
            used_buckets=used_buckets,
            fill_rate=fill_rate,
            max_bucket_size=max_bucket,
            recommended_n_features=recommended
        )

    def _recommend_size(self, collision_rate: float, fill_rate: float) -> int:
        """Recommande une taille optimale"""
        if collision_rate > 0.15:
            # Beaucoup de collisions → doubler
            return min(self.n_features * 2, 65536)
        elif collision_rate > 0.10:
            # Collisions modérées → augmenter 50%
            return min(int(self.n_features * 1.5), 65536)
        elif fill_rate < 0.05 and collision_rate < 0.02:
            # Très peu utilisé et peu de collisions → réduire
            return max(self.n_features // 2, 2048)

        return self.n_features

    def print_report(self):
        """Affiche un rapport détaillé"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("📊 COLLISION REPORT")
        print("=" * 60)
        print(f"  Configuration:         n_features = {self.n_features:,}")
        print(f"  Total insertions:      {stats.total_tokens:,}")
        print(f"  Unique tokens:         {stats.unique_tokens:,}")
        print(f"  Collisions:            {stats.collision_count:,}")
        print(f"  Collision rate:        {stats.collision_rate * 100:.2f}%")
        print(f"  Buckets used:          {stats.used_buckets:,} / {self.n_features:,}")
        print(f"  Fill rate:             {stats.fill_rate * 100:.2f}%")
        print(f"  Max bucket size:       {stats.max_bucket_size}")
        print("-" * 60)

        # Recommandation
        if stats.recommended_n_features != self.n_features:
            print(f"  ⚠️  RECOMMANDATION: Changer n_features à {stats.recommended_n_features:,}")
        else:
            print(f"  ✅ Configuration actuelle optimale")

        # Distribution des buckets
        print("\n  Distribution des collisions:")
        size_counts = defaultdict(int)
        for tokens in self.bucket_tokens.values():
            size_counts[len(tokens)] += 1

        for size in sorted(size_counts.keys())[:10]:
            count = size_counts[size]
            bar = "█" * min(count // 10, 30)
            print(f"    {size} token(s): {count:5} buckets {bar}")

        print("=" * 60 + "\n")

    def get_worst_collisions(self, top_n: int = 10) -> List[tuple]:
        """Retourne les buckets avec le plus de collisions"""
        bucket_list = [
            (idx, list(tokens))
            for idx, tokens in self.bucket_tokens.items()
            if len(tokens) > 1
        ]

        bucket_list.sort(key=lambda x: -len(x[1]))

        return bucket_list[:top_n]

"""
hash_utils.py - Version utilisant sklearn directement
"""

import numpy as np
from typing import Tuple, List

# Essayer d'importer le hash sklearn directement
try:
    from sklearn.utils.murmurhash import murmurhash3_bytes_s32
    HAS_SKLEARN_HASH = True
except ImportError:
    HAS_SKLEARN_HASH = False
    import mmh3


def hash_token(token: str, seed: int = 0) -> Tuple[int, int]:
    """
    Hash un token et retourne (hash_unsigned, hash_signed).
    Utilise exactement la même fonction que sklearn.
    """
    token_bytes = token.encode('utf-8')

    if HAS_SKLEARN_HASH:
        # Utiliser exactement la même fonction que sklearn
        h_signed = murmurhash3_bytes_s32(token_bytes, seed=seed)
        h_unsigned = h_signed & 0xFFFFFFFF
    else:
        # Fallback mmh3
        h_signed = mmh3.hash(token_bytes, seed=seed, signed=True)
        h_unsigned = mmh3.hash(token_bytes, seed=seed, signed=False)

    return h_unsigned, h_signed


def hash_to_index_and_sign(token: str, n_features: int) -> Tuple[int, int]:
    """
    Convertit un token en index et signe, 100% compatible sklearn.
    """
    h_unsigned, h_signed = hash_token(token, seed=0)

    # Sklearn utilise abs(h_signed) % n_features pour les indices
    idx = abs(h_signed) % n_features
    sign = 1 if h_signed >= 0 else -1

    return int(idx), sign


def verify_sklearn_compatibility(n_features: int = 8192, verbose: bool = True) -> bool:
    """
    Vérifie que notre implémentation match sklearn HashingVectorizer.
    """
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
    except ImportError:
        print("⚠️  sklearn non disponible, skip vérification")
        return True

    test_tokens = [
        "C_<test",
        "W_bonjour",
        "B_hello_world",
        "T_a_b_c",
        "POS_START_test",
        "BPE_##ing",
        "C_été",
        "W_123",
        "test",
        "a",
        "hello",
        "world",
        "supercalifragilisticexpialidocious",
        "",  # string vide
        " ",  # espace
        "hello world",  # avec espace
    ]

    # Créer le vectorizer sklearn comme référence
    sklearn_vec = HashingVectorizer(
        n_features=n_features,
        alternate_sign=True,
        norm=None,
        analyzer=lambda x: [x]  # Un seul token = le string entier
    )

    all_match = True

    if verbose:
        print(f"🔍 Vérification compatibilité sklearn:")
        print(f"   Utilise sklearn.murmurhash: {HAS_SKLEARN_HASH}")
        print("-" * 70)

    for token in test_tokens:
        # Notre implémentation
        our_idx, our_sign = hash_to_index_and_sign(token, n_features)

        # sklearn
        result = sklearn_vec.transform([token])

        if result.nnz == 0:
            # Token vide ou autre cas spécial
            sk_idx, sk_sign = -1, 0
        else:
            sk_idx = result.indices[0]
            sk_sign = 1 if result.data[0] > 0 else -1

        match = (our_idx == sk_idx) and (our_sign == sk_sign)

        if verbose:
            status = "✅" if match else "❌"
            token_display = repr(token) if len(token) < 30 else repr(token[:27] + "...")
            print(f"  {status} {token_display:40s} | ours: ({our_idx:5d}, {our_sign:+d}) | sklearn: ({sk_idx:5d}, {sk_sign:+d})")

        if not match:
            all_match = False
            if verbose:
                # Debug supplémentaire pour les erreurs
                h_u, h_s = hash_token(token, seed=0)
                print(f"      DEBUG: hash_unsigned={h_u}, hash_signed={h_s}")

    if verbose:
        print("-" * 70)
        if all_match:
            print("✅ Tous les tests passent!")
        else:
            print("❌ Incompatibilités détectées!")

    return all_match


# ============================================================================
# Implémentation Python pure pour C++ (référence)
# ============================================================================

def murmurhash3_32_reference(key: bytes, seed: int = 0) -> int:
    """
    MurmurHash3 32-bit - Implémentation de référence.
    À utiliser pour générer le code C++.
    """
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    mask = 0xFFFFFFFF

    h = seed & mask
    length = len(key)

    # Process 4-byte chunks
    n_blocks = length // 4

    for i in range(n_blocks):
        # Little-endian
        k = (key[i*4] |
             (key[i*4 + 1] << 8) |
             (key[i*4 + 2] << 16) |
             (key[i*4 + 3] << 24))

        k = (k * c1) & mask
        k = ((k << 15) | (k >> 17)) & mask  # rotl32
        k = (k * c2) & mask

        h ^= k
        h = ((h << 13) | (h >> 19)) & mask  # rotl32
        h = ((h * 5) + 0xe6546b64) & mask

    # Process remaining bytes
    tail_index = n_blocks * 4
    k1 = 0
    tail_size = length & 3

    if tail_size >= 3:
        k1 ^= key[tail_index + 2] << 16
    if tail_size >= 2:
        k1 ^= key[tail_index + 1] << 8
    if tail_size >= 1:
        k1 ^= key[tail_index]
        k1 = (k1 * c1) & mask
        k1 = ((k1 << 15) | (k1 >> 17)) & mask
        k1 = (k1 * c2) & mask
        h ^= k1

    # Finalization
    h ^= length

    # fmix32
    h ^= (h >> 16)
    h = (h * 0x85ebca6b) & mask
    h ^= (h >> 13)
    h = (h * 0xc2b2ae35) & mask
    h ^= (h >> 16)

    return h


def generate_cpp_hash_impl() -> str:
    """
    Génère l'implémentation C++ de MurmurHash3.
    """
    return '''
// MurmurHash3 32-bit implementation
// Compatible with sklearn HashingVectorizer

#include <cstdint>
#include <cstring>

inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

uint32_t murmurhash3_32(const uint8_t* key, size_t len, uint32_t seed = 0) {
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    
    uint32_t h = seed;
    
    // Body - process 4-byte blocks
    const size_t nblocks = len / 4;
    const uint32_t* blocks = (const uint32_t*)(key);
    
    for (size_t i = 0; i < nblocks; i++) {
        uint32_t k = blocks[i];  // Assumes little-endian
        
        k *= c1;
        k = rotl32(k, 15);
        k *= c2;
        
        h ^= k;
        h = rotl32(h, 13);
        h = h * 5 + 0xe6546b64;
    }
    
    // Tail - process remaining bytes
    const uint8_t* tail = key + nblocks * 4;
    uint32_t k1 = 0;
    
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16; [[fallthrough]];
        case 2: k1 ^= tail[1] << 8;  [[fallthrough]];
        case 1: k1 ^= tail[0];
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;
                h ^= k1;
    }
    
    // Finalization
    h ^= len;
    h = fmix32(h);
    
    return h;
}

// Helper pour strings
inline uint32_t hashString(const char* str, uint32_t seed = 0) {
    return murmurhash3_32((const uint8_t*)str, strlen(str), seed);
}

// Conversion en index et signe (compatible sklearn alternate_sign=True)
inline void hashToIndexAndSign(const char* token, uint32_t nFeatures, 
                                uint32_t& outIndex, int8_t& outSign) {
    int32_t h = (int32_t)hashString(token, 0);
    // sklearn: index = abs(h) % nFeatures
    uint32_t abs_h = (h >= 0) ? (uint32_t)h : (uint32_t)-(int64_t)h;
    outIndex = abs_h % nFeatures;
    // sklearn: signe basé sur le bit de signe en int32
    outSign = (h >= 0) ? 1 : -1;
}
'''


def generate_cpp_test_vectors(n_features: int = 8192) -> str:
    """
    Génère des vecteurs de test C++ pour validation.
    """
    test_cases = []

    # Tokens de test
    tokens = [
        ("test", "simple"),
        ("hello", "simple2"),
        ("W_bonjour", "word_prefix"),
        ("C_ab", "char_ngram"),
        ("B_the_cat", "bigram"),
        ("a", "single_char"),
        ("123", "numbers"),
    ]

    cpp_code = f'''
// Auto-generated test vectors
// Generated for n_features = {n_features}

struct HashTest {{
    const char* token;
    uint32_t expectedIndex;
    int8_t expectedSign;
    const char* description;
}};

const HashTest HASH_TESTS[] = {{
'''

    for token, desc in tokens:
        idx, sign = hash_to_index_and_sign(token, n_features)
        h_u, h_s = hash_token(token, seed=0)
        cpp_code += f'    {{"{token}", {idx}, {sign}, "{desc}"}},  // hash=0x{h_u:08x}\n'
        test_cases.append((token, idx, sign))

    cpp_code += f'''
}};

const size_t NUM_HASH_TESTS = sizeof(HASH_TESTS) / sizeof(HASH_TESTS[0]);
const uint32_t HASH_N_FEATURES = {n_features};

bool runHashTests() {{
    Serial.println("\\n=== MurmurHash3 Verification ===");
    bool allPass = true;
    
    for (size_t i = 0; i < NUM_HASH_TESTS; i++) {{
        const auto& t = HASH_TESTS[i];
        
        uint32_t idx;
        int8_t sign;
        hashToIndexAndSign(t.token, HASH_N_FEATURES, idx, sign);
        
        bool pass = (idx == t.expectedIndex) && (sign == t.expectedSign);
        allPass &= pass;
        
        Serial.printf("%s %-20s: idx=%5u (exp %5u), sign=%+d (exp %+d)\\n",
                      pass ? "PASS" : "FAIL",
                      t.token,
                      idx, t.expectedIndex,
                      sign, t.expectedSign);
    }}
    
    if (allPass) {{
        Serial.println("\\n✓ All hash tests PASSED!");
    }} else {{
        Serial.println("\\n✗ Hash tests FAILED - implementation mismatch!");
    }}
    
    return allPass;
}}
'''

    return cpp_code


if __name__ == "__main__":
    print("="*70)
    print("Hash Utils - Test")
    print("="*70)
    print(f"Using sklearn murmurhash: {HAS_SKLEARN_HASH}")
    print()

    success = verify_sklearn_compatibility(verbose=True)

    if success:
        print("\n\nGenerating C++ code...")
        print(generate_cpp_hash_impl()[:500] + "...")

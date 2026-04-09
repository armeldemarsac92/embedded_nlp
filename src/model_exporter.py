"""
Export des modèles entraînés vers C++ (float32 et int8).
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class QuantizationInfo:
    """Informations de quantification par couche"""
    scale: float
    zero_point: int = 0  # Symétrique, donc 0


def quantize_symmetric(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Quantification symétrique INT8.

    Maps [-max_abs, +max_abs] → [-127, +127]
    """
    max_abs = np.max(np.abs(weights))

    if max_abs < 1e-10:
        return np.zeros_like(weights, dtype=np.int8), 1.0

    scale = max_abs / 127.0
    quantized = np.round(weights / scale).astype(np.int32)
    quantized = np.clip(quantized, -127, 127).astype(np.int8)

    return quantized, float(scale)


def compute_quantization_error(original: np.ndarray, quantized: np.ndarray, scale: float) -> dict:
    """Calcule l'erreur de quantification"""
    reconstructed = quantized.astype(np.float32) * scale
    error = np.abs(original - reconstructed)

    return {
        'max_error': float(np.max(error)),
        'mean_error': float(np.mean(error)),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'relative_error': float(np.mean(error) / (np.mean(np.abs(original)) + 1e-10))
    }


class ModelExporter:
    """
    Exporte un modèle sklearn vers C++ (float32 et/ou int8).
    """

    def __init__(
            self,
            classifier,
            params: dict,
            categories: List[str],
            n_features: int = 8192
    ):
        self.clf = classifier
        self.params = params
        self.categories = categories
        self.n_features = n_features

        # Dimensions
        self.hidden1_size = classifier.coefs_[0].shape[1]
        self.hidden2_size = classifier.coefs_[1].shape[1]
        self.output_size = classifier.coefs_[2].shape[1]

        # Activation
        self.activation = getattr(classifier, 'activation', 'relu')

    def export_float32(self, filepath: str):
        """
        Exporte le modèle en float32 (compatible code actuel).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(filepath, 'w') as f:
            self._write_header(f, timestamp, "float32")
            self._write_dimensions(f)
            self._write_feature_params(f)
            self._write_categories(f)
            self._write_activation(f)

            # Poids en float32
            self._write_float_matrix(f, "W1", self.clf.coefs_[0].T,
                                     f"[{self.hidden1_size}][{self.n_features}]")
            self._write_float_vector(f, "b1", self.clf.intercepts_[0])

            self._write_float_matrix(f, "W2", self.clf.coefs_[1].T,
                                     f"[{self.hidden2_size}][{self.hidden1_size}]")
            self._write_float_vector(f, "b2", self.clf.intercepts_[1])

            self._write_float_matrix(f, "W3", self.clf.coefs_[2].T,
                                     f"[{self.output_size}][{self.hidden2_size}]")
            self._write_float_vector(f, "b3", self.clf.intercepts_[2])

            self._write_memory_stats(f, quantized=False)
            self._write_footer(f)

        print(f"✅ Float32 model exported: {filepath}")

    def export_int8(self, filepath: str) -> Dict[str, QuantizationInfo]:
        """
        Exporte le modèle en INT8 quantifié.

        Returns:
            Dictionnaire des informations de quantification
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Quantifier toutes les couches
        W1_q, W1_scale = quantize_symmetric(self.clf.coefs_[0])
        W2_q, W2_scale = quantize_symmetric(self.clf.coefs_[1])
        W3_q, W3_scale = quantize_symmetric(self.clf.coefs_[2])

        b1_q, b1_scale = quantize_symmetric(self.clf.intercepts_[0])
        b2_q, b2_scale = quantize_symmetric(self.clf.intercepts_[1])
        b3_q, b3_scale = quantize_symmetric(self.clf.intercepts_[2])

        # Rapport d'erreur
        print("\n📊 Quantization Error Analysis:")
        for name, orig, quant, scale in [
            ("W1", self.clf.coefs_[0], W1_q, W1_scale),
            ("W2", self.clf.coefs_[1], W2_q, W2_scale),
            ("W3", self.clf.coefs_[2], W3_q, W3_scale),
        ]:
            err = compute_quantization_error(orig, quant, scale)
            print(f"   {name}: max={err['max_error']:.6f}, rel={err['relative_error'] * 100:.2f}%")

        with open(filepath, 'w') as f:
            self._write_header(f, timestamp, "int8 quantized")
            self._write_dimensions(f)
            self._write_feature_params(f)
            self._write_categories(f)
            self._write_activation(f)

            # Scales de quantification
            f.write("// Quantization scales\n")
            f.write(f"const float W1_SCALE = {W1_scale}f;\n")
            f.write(f"const float W2_SCALE = {W2_scale}f;\n")
            f.write(f"const float W3_SCALE = {W3_scale}f;\n")
            f.write(f"const float B1_SCALE = {b1_scale}f;\n")
            f.write(f"const float B2_SCALE = {b2_scale}f;\n")
            f.write(f"const float B3_SCALE = {b3_scale}f;\n\n")

            # Poids en int8
            self._write_int8_matrix(f, "W1", W1_q.T,
                                    f"[{self.hidden1_size}][{self.n_features}]")
            self._write_int8_vector(f, "b1", b1_q)

            self._write_int8_matrix(f, "W2", W2_q.T,
                                    f"[{self.hidden2_size}][{self.hidden1_size}]")
            self._write_int8_vector(f, "b2", b2_q)

            self._write_int8_matrix(f, "W3", W3_q.T,
                                    f"[{self.output_size}][{self.hidden2_size}]")
            self._write_int8_vector(f, "b3", b3_q)

            self._write_memory_stats(f, quantized=True)
            self._write_footer(f)

        print(f"✅ INT8 model exported: {filepath}")

        return {
            'W1': QuantizationInfo(W1_scale),
            'W2': QuantizationInfo(W2_scale),
            'W3': QuantizationInfo(W3_scale),
            'b1': QuantizationInfo(b1_scale),
            'b2': QuantizationInfo(b2_scale),
            'b3': QuantizationInfo(b3_scale),
        }

    def export_verification_code(self, filepath: str, test_texts: List[str]):
        """
        Génère du code C++ pour vérifier la compatibilité.
        """
        with open(filepath, 'w') as f:
            f.write("// Verification code - Auto-generated\n")
            f.write("// Compare Python and C++ outputs\n\n")

            f.write('#include "NlpManager.h"\n\n')

            f.write("void runVerificationTests() {\n")
            f.write('    Serial.println("=== VERIFICATION TESTS ===");\n\n')

            for i, text in enumerate(test_texts[:5]):
                escaped = text.replace('\\', '\\\\').replace('"', '\\"')
                f.write(f'    // Test {i + 1}\n')
                f.write(f'    NlpManager::getInstance().debugPrediction("{escaped}");\n\n')

            f.write('    Serial.println("=== END TESTS ===");\n')
            f.write("}\n")

        print(f"✅ Verification code exported: {filepath}")

    # === Méthodes privées d'écriture ===

    def _write_header(self, f, timestamp: str, model_type: str):
        f.write(f"// Neural Network Model for Teensy 4.1\n")
        f.write(f"// Type: {model_type}\n")
        f.write(f"// Generated: {timestamp}\n")
        f.write(f"// Activation: {self.activation}\n\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("#include <Arduino.h>\n\n")

    def _write_footer(self, f):
        f.write("#endif // MODEL_WEIGHTS_H\n")

    def _write_dimensions(self, f):
        f.write("// Network dimensions\n")
        f.write(f"const int INPUT_SIZE = {self.n_features};\n")
        f.write(f"const int HIDDEN1_SIZE = {self.hidden1_size};\n")
        f.write(f"const int HIDDEN2_SIZE = {self.hidden2_size};\n")
        f.write(f"const int OUTPUT_SIZE = {self.output_size};\n\n")

    def _write_feature_params(self, f):
        f.write("// Feature extraction parameters\n")
        f.write("namespace FeatureParams {\n")

        for key, val in self.params.items():
            if isinstance(val, float):
                f.write(f"    const float {key} = {val}f;\n")
            elif isinstance(val, int):
                f.write(f"    const int {key} = {val};\n")
            elif isinstance(val, str):
                f.write(f'    const char* {key} = "{val}";\n')

        f.write("}\n\n")

    def _write_categories(self, f):
        f.write(f"// Output categories\n")
        f.write(f"const char* CATEGORIES[{self.output_size}] = {{\n")
        for cat in self.categories:
            f.write(f'    "{cat}",\n')
        f.write("};\n\n")

    def _write_activation(self, f):
        f.write(f"// Activation function\n")
        f.write(f"#define ACTIVATION_{self.activation.upper()} 1\n\n")

    def _write_float_matrix(self, f, name: str, matrix: np.ndarray, shape_comment: str):
        rows, cols = matrix.shape
        f.write(f"// {name}: {shape_comment}\n")
        f.write(f"const float {name}[{rows}][{cols}] PROGMEM = {{\n")

        for row in matrix:
            f.write("    {")
            f.write(", ".join(f"{v:.8f}f" for v in row))
            f.write("},\n")

        f.write("};\n\n")

    def _write_float_vector(self, f, name: str, vector: np.ndarray):
        f.write(f"const float {name}[{len(vector)}] PROGMEM = {{\n    ")
        f.write(", ".join(f"{v:.8f}f" for v in vector))
        f.write("\n};\n\n")

    def _write_int8_matrix(self, f, name: str, matrix: np.ndarray, shape_comment: str):
        rows, cols = matrix.shape
        f.write(f"// {name}: {shape_comment}\n")
        f.write(f"const int8_t {name}[{rows}][{cols}] PROGMEM = {{\n")

        for row in matrix:
            f.write("    {")
            f.write(", ".join(str(v) for v in row))
            f.write("},\n")

        f.write("};\n\n")

    def _write_int8_vector(self, f, name: str, vector: np.ndarray):
        f.write(f"const int8_t {name}[{len(vector)}] PROGMEM = {{\n    ")
        f.write(", ".join(str(v) for v in vector))
        f.write("\n};\n\n")

    def _write_memory_stats(self, f, quantized: bool):
        total_weights = (
                self.clf.coefs_[0].size + self.clf.coefs_[1].size + self.clf.coefs_[2].size +
                self.clf.intercepts_[0].size + self.clf.intercepts_[1].size + self.clf.intercepts_[2].size
        )

        if quantized:
            memory_bytes = total_weights  # 1 byte per param
            memory_str = f"{memory_bytes:,} bytes ({memory_bytes / 1024:.2f} KB)"
        else:
            memory_bytes = total_weights * 4  # 4 bytes per float
            memory_str = f"{memory_bytes:,} bytes ({memory_bytes / 1024:.2f} KB)"

        f.write(f"// Memory usage: {memory_str}\n")
        f.write(f"// Total parameters: {total_weights:,}\n")
        if quantized:
            f.write(f"// Compression: 4x vs float32\n")
        f.write("\n")

# vbmlr/predict.py
import os
import numpy as np
import math

def _cal_prob_cpp(sm: float) -> float:
    # exp(-sm)/(1+exp(-sm)) = sigmoid(-sm)
    if sm >= 0:
        e = math.exp(-sm)
        return e / (1.0 + e)
    else:
        e = math.exp(sm)
        return 1.0 / (1.0 + e)

def _load_vector_csv(path: str):
    if not os.path.isfile(path):
        return None
    txt = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
    if not txt:
        return None
    toks = [t for t in txt.replace("\n", ",").split(",") if t.strip() != ""]
    # atof; ignore no-numeric tokens
    out = []
    for t in toks:
        try:
            out.append(float(t))
        except:
            pass
    return np.array(out, dtype=np.float64) if len(out) > 0 else None

class VBMLRPredictOneVsOne:
    """
      - get disc_{j}_{i}.txt
      - sm = dot(w[:8], feats[:8]) + bias(w[8])
      - SH = sigmoid(-sm)
      - vote i<j vs i>j
    """
    def __init__(self, base_dir: str, nb_classes: int = 9):
        self.base_dir = base_dir
        self.C = nb_classes
        self.disc = [[None for _ in range(self.C)] for _ in range(self.C)]
        self.reload()

    def reload(self):
        for j in range(1, self.C + 1):
            for i in range(1, self.C + 1):
                if i == j:
                    continue
                path = os.path.join(self.base_dir, f"disc_{j}_{i}.txt")
                self.disc[j-1][i-1] = _load_vector_csv(path)

    @staticmethod
    def _cal_sm_cpp(w: np.ndarray, feats: np.ndarray) -> float:
        d = min(8, len(feats), len(w) - 1)
        sm = float(np.dot(w[:d], feats[:d]) + w[8])  # bias = w[8]
        return sm

    @staticmethod
    def _dec_cpp(x: float) -> int:
        return 1 if (x > 0.5) else 0

    def predict(self, feats8: np.ndarray) -> int:
        feats8 = np.asarray(feats8, dtype=np.float64).reshape(-1)
        vectorC = []

        for j in range(1, self.C + 1):
            prob_C = 0.0
            for i in range(1, self.C + 1):
                if i == j:
                    continue
                w = self.disc[j-1][i-1]
                if w is None or len(w) < 9:
                    continue

                sm = self._cal_sm_cpp(w, feats8)
                SH = _cal_prob_cpp(sm)

                if i < j:
                    prob_C += (1 - self._dec_cpp(1 - SH))
                else:
                    prob_C += self._dec_cpp(SH)

            vectorC.append(prob_C)

        return int(np.argmax(vectorC)) + 1

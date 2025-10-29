
import json
import math

def load_metrics():
    with open("results/metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)

def test_faithfulness():
    m = load_metrics()
    assert not math.isnan(m["faithfulness_mean"]), "Faithfulness не рассчитан"
    assert m["faithfulness_mean"] >= 0.7, f"❌ faithfulness_mean={m['faithfulness_mean']:.3f} < 0.7"

def test_rouge():
    m = load_metrics()
    assert m["rougeL"] >= 0.2, f"rougeL={m['rougeL']:.3f} < 0.2"

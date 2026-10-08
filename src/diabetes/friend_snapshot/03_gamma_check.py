"""03 - regression test: both gamma implementations reproduce the hand-worked toy example."""
import pandas as pd
from rst import gamma_vprs, encode, granules, gamma_fast

toy = pd.DataFrame({"HighBP": [1, 1, 1, 0, 0, 0], "Smoker": [1, 1, 0, 0, 1, 1],
                    "Diabetes": [1, 1, 0, 0, 1, 0]})
B, y = ["HighBP", "Smoker"], toy["Diabetes"].to_numpy(float)
gid = granules(encode(toy, B), B, len(toy))
for beta, want in [(1.0, 0.6667), (0.6, 0.6667), (0.5, 1.0)]:
    ref = round(gamma_vprs(toy, B, "Diabetes", beta), 4)
    fast = round(gamma_fast(gid, y, beta), 4)
    assert ref == fast == want, (beta, ref, fast, want)
    print(f"beta={beta}: reference {ref}, fast {fast}, expected {want}  OK")

from pathlib import Path
# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------
# Format (same as the authors' repository):
#   m
#   n
#   p[1][1] ... p[1][n]        processing time of each job on machine 1
#   ...
#   p[m][1] ... p[m][n]
#   tau[0][0] ... tau[0][m+1]  travel time between each pair of stations
#   ...
#   tau[m+1][0] ... tau[m+1][m+1]


def read_instance(path):
    tokens = Path(path).read_text().split()
    it = iter(int(tok) for tok in tokens)
    m = next(it)
    n = next(it)
    # p is padded to (m+2) x (n+1): row 0 (input buffer) and row m+1 (output buffer)
    # are zero, and column 0 (job index 0 = "no job") is zero, so that p[i, x_i]
    # is always a valid table lookup.
    p = [[0] * (n + 1) for _ in range(m + 2)]
    for i in range(1, m + 1):
        for k in range(1, n + 1):
            p[i][k] = next(it)
    tau = [[next(it) for _ in range(m + 2)] for _ in range(m + 2)]
    return m, n, p, tau

from math import floor, ceil, comb, copysign
import random
import warnings
import numpy as np


RHS_CACHE = dict[tuple[int, int, int, int], float]()
LHS_CACHE = dict[tuple[int, int, int, int], float]()
COMB_ALLOWED_CACHE = dict[tuple[int, int, int, int], float]()


class ABCDH_params:
    """
        ABCDHParams
    A structure holding parameters for ABCD graph generator. Fields:
    * is_simple:bool            if hypergraph is simple
    * w:list[int]               list of vertex degrees
    * y:list[int]               community degree
    * z:list[int]               network degree
    * s:list[int]               list of cluster sizes
    * x:float | None            background graph fraction
    * q:list[float]             distribution of hyperedge sizes
    * wcd:ndarray[float, float] desired composition of hyperedges
    * maxiter:int               maximum number of iterations trying to resolve collisions per collision
    """

    def __init__(self, w, s, x, q, wcd, is_simple, maxiter):
        self.w = w
        self.s = s
        self.x = x
        self.q = q
        self.wcd = wcd
        self.is_simple = is_simple
        self.maxiter = maxiter
        self.y = np.full(len(w), -1, dtype=int)
        self.z = np.full(len(w), -1, dtype=int)

        if len(w) != sum(s):
            warnings.warn(f"inconsistent data w {len(w)} and s {sum(s)}")
        if not all(_ >= 0 for _ in w):
            warnings.warn("negative degree passed")
        if not all(_ >= 0 for _ in s):
            warnings.warn("negative community size passed")
        if not all(_ >= 0 for _ in q):
            warnings.warn("negative hyperedge proportion passed")
        if not np.all(wcd >= 0):
            warnings.warn("negative hyperedge composition passed")
        if not 0 <= x <= 1:
            warnings.warn(f"x={x} not in [0,1] interval")

        sq = sum(q)
        if sq != 1:
            warnings.warn(
                f"distribution of hyperedge proportions {sq} does not add up to 1. Fixing."
            )

            if sq < np.finfo(np.float32).eps:
                warnings.warn("sum of hyperedge proportions is abnormally small")
            q = [q_ / sq for q_ in q]

        if np.shape(wcd) != (len(q), len(q)):
            warnings.warn("incorrect dimension of hyperedge composition matrix")

        for d in range(len(q)):
            for c in range(d // 2):
                if wcd[c, d] != 0:
                    warnings.warn(
                        f"weight for c <= d/2 is positive where c={c} and d={d}. Fixing."
                    )
                    wcd[c, d] = 0.0
            for c in range(d + 1, len(q)):
                if wcd[c, d] != 0:
                    warnings.warn(
                        f"weight for c > d is positive where c={c} and d={d}. Fixing."
                    )
                    wcd[c, d] = 0.0

            swcd = sum(wcd[:, d])
            if swcd != 1:
                warnings.warn(
                    f"distribution of hyperedge composition for d={d} does not add up to 1. Fixing."
                )
            if swcd < np.finfo(np.float32).eps:
                warnings.warn(
                    f"sum of hyperedge composition {swcd} is abnormally small for d={d}"
                )
                w[:, d] = [i / swcd for i in w[:, d]]


def get_rhs(n, cj, d, c):
    if (n, cj, d, c) not in RHS_CACHE:
        RHS_CACHE[(n, cj, d, c)] = comb(cj, c) * comb(n - cj, d - c)
    return RHS_CACHE[(n, cj, d, c)]


def get_lhs(params, n, cj, d, c):
    sum_ = sum(range(d // 2 + 1, c))
    if (n, cj, d, c) not in LHS_CACHE:
        LHS_CACHE[(n, cj, d, c)] = (
            params.q[d]
            * params.wcd[sum_, d]
            * comb(d - sum_, c - sum_)
            * (cj / n) ** (c - sum_)
            * (1 - cj / n) ** (d - c)
        )
    return LHS_CACHE[(n, cj, d, c)]


def get_comballowed(cj, yi, zi, params, n):
    for d in range(1, params.wcd.shape[1]):
        for c in range((d + 1) // 2, d + 1):
            xi = yi * get_lhs(params, n, cj, d, c) + zi * (
                params.q[d] * comb(d, c) * (cj / n) ** c * (1 - cj / n) ** (d - c)
            )
            if xi > get_rhs(n, cj, d, c):
                return False
    return True


def randround(x):
    d = floor(x)
    return d + (random.random() < x - d)


def generate_1hyperedges(params):
    n = len(params.w)
    ww = params.w
    if params.is_simple:
        m1 = min(randround(params.q[0] * sum(params.w)), n)
        he1 = random.choices(range(n), k=m1, weights=ww)
        assert len(he1) == len(np.unique(he1))
    else:
        m1 = randround(params.q[0] * sum(params.w))
        he1 = random.choices(range(n), k=m1, weights=ww)

    for i in he1:
        assert params.w[i] > 0
        params.w[i] -= 1
    return [[i] for i in he1]


def populate_clusters(params):
    # Note that populate_clusters is not thread safe
    RHS_CACHE.clear()
    LHS_CACHE.clear()
    COMB_ALLOWED_CACHE.clear()

    n = len(params.w)
    clusters = np.full(n, -1, dtype=int)
    slots = np.copy(params.s)
    community_allowed = np.full(len(slots), True)

    for i in sorted(range(len(params.w)), key=lambda k: params.w[k], reverse=True):
        loc = -1
        wts_all = slots
        for _ in range(10):
            pos = random.choice(range(len(wts_all)))
            cj = params.s[pos]
            choice_allowed = get_comballowed(cj, params.y[i], params.z[i], params, n)
            if choice_allowed:
                loc = pos
                break

        if loc == -1:
            community_allowed.fill(True)
            for j in range(len(slots)):
                cj = params.s[j]
                community_allowed[j] = get_comballowed(
                    cj, params.y[i], params.z[i], params, n
                )
                comm_idxs = np.nonzero(community_allowed)
                wts = slots[comm_idxs]
                if sum(wts) == 0:
                    warnings.warn(
                        "hypergraph is too tight. Failed to find community for node $i"
                    )
                loc = random.choices(comm_idxs, k=1, weights=wts)
        clusters[i] = loc
        slots[loc] -= 1

    assert sum(slots) == 0
    assert (min(clusters), max(clusters)) == (0, len(params.s) - 1)
    return clusters


def config_model(clusters, params, he1):
    L = len(params.q)

    edges = list[list[int]]()

    # community graphs
    community_stumps = list[int]()  # stumps left for filling d-c slots
    edges_with_missing_size = list[
        tuple[[], int]
    ]()  # partial edges with missing d-c slots

    for j in range(len(params.s)):
        cluster_idxs = np.where(clusters == j)[0]

        md = np.zeros(L, dtype=int)
        pj = sum(params.y[i] for i in cluster_idxs)
        for d in range(L - 1, 0, -1):
            sumq2 = sum(params.q[1 : d + 1])
            if sumq2 > 0:
                new_md = [
                    range(d + 1, L)[i] * md[d + 1 : L][i]
                    for i in range(len(md[d + 1 : L]))
                ]
                md[d] = floor(params.q[d] / sumq2 * (pj - sum(new_md)) / d)

        sumpj = sum(d * md[d] for d in range(L))
        if pj > sumpj:
            print(f"Moving {pj - sumpj} stumps from community {j} to background graph")

        while pj > sumpj:
            dec_idx = random.choices(
                range(len(cluster_idxs)),
                k=1,
                weights=[params.y[i] for i in cluster_idxs],
            )[0]
            params.y[dec_idx] -= 1
            params.z[dec_idx] += 1
            pj -= 1

        assert pj == sumpj

        if pj == 0:
            assert sum(params.y[i] for i in cluster_idxs) == 0
        else:
            mcd = np.zeros((L, L), dtype=int)
            for d in range(1, L):
                for c in range(d, (d - 1) // 2, -1):
                    sumwfd = sum(params.wcd[d // 2 : c + 1, d])
                    if sumwfd > 0:
                        mcd[c, d] = randround(
                            params.wcd[c, d] / sumwfd * (md[d] - sum(mcd[c : d + 1, d]))
                        )

                assert md[d] == sum(mcd[:, d])

            assert (
                sum(
                    d * mcd[c, d]
                    for d in range(1, L)
                    for c in range(d, (d - 1) // 2, -1)
                )
                == pj
            )

            pjc = sum(
                c * mcd[c, d] for d in range(1, L) for c in range(d, (d - 1) // 2, -1)
            )

            yc = [params.y[i] * pjc / pj for i in cluster_idxs]
            yc_base = [floor(yc_) for yc_ in yc]
            yc_rem = [yc[i] - yc_base[i] for i in range(len(yc))]
            tail_size = pjc - sum(yc_base)

            assert 0 <= tail_size <= len(yc_rem)
            if tail_size > 0:
                normalized_weights = [y_r / sum(yc_rem) for y_r in yc_rem]
                additional_points = np.random.choice(
                    range(len(yc_rem)), tail_size, p=normalized_weights, replace=False
                )
                for point in additional_points:
                    yc_base[point] += 1

            assert sum(yc_base) == pjc
            assert [floor(_) for _ in yc] <= yc_base <= [ceil(_) for _ in yc]

            assert len(cluster_idxs) == len(yc_base)

            internal_stumps = list[int]()
            for yci, index in zip(yc_base, list(cluster_idxs)):
                internal_stumps.extend([index for _ in range(yci)])

            random.shuffle(internal_stumps)
            assert np.sum(mcd) <= len(internal_stumps)
            assert pjc == len(internal_stumps)

            stump_idx = 0
            for d in range(1, L):
                for c in range((d + 1) // 2, d + 1):
                    for _ in range(mcd[c, d]):
                        stump_end = stump_idx + c
                        edges_with_missing_size.append(
                            (internal_stumps[stump_idx : stump_end + 1], d - c)
                        )
                        stump_idx += c

            assert stump_idx == len(internal_stumps)

            start_len = len(community_stumps)
            y_cluster_idxs = [params.y[i] for i in cluster_idxs]
            y_cluster_idxs_less_yc_base = [
                y_cluster_idxs[i] - yc_base[i] for i in range(len(y_cluster_idxs))
            ]
            for yri, index in zip(y_cluster_idxs_less_yc_base, cluster_idxs):
                community_stumps.extend([index for _ in range(yri)])

            end_len = len(community_stumps)
            assert end_len - start_len == pj - pjc

    random.shuffle(community_stumps)

    stump_idx = 0
    for he, dc in edges_with_missing_size:
        if dc > 0:
            stump_end = stump_idx + dc
            he.extend(community_stumps[stump_idx : stump_end + 1])
            stump_idx += dc
        edges.append(sorted(he))

    # background graph

    # stumps transferred from community graphs to background graph
    background_stumps = community_stumps[stump_idx:]

    if len(background_stumps):
        print(
            f"moved {len(background_stumps)} non-matched stumps from community graphs to background graph"
        )

    md = np.zeros(L, dtype=int)
    p = sum(params.z) + len(background_stumps)
    for d in range(L - 1, 0, -1):
        sumq2 = sum(params.q[1 : d + 1])
        if sumq2 > 0:
            new_md = [
                range(d + 1, L)[i] * md[d + 1 : L][i] for i in range(len(md[d + 1 : L]))
            ]
            md[d] = floor(params.q[d] / sumq2 * (p - sum(new_md)) / d)

    for index, value in enumerate(params.z):
        background_stumps.extend(np.full(value, index))

    random.shuffle(background_stumps)

    assert sum(md) <= len(background_stumps)
    stump_idx = 0
    for d in range(1, L):
        for _ in range(md[d]):
            stump_end = stump_idx + d
            edges.append(sorted(background_stumps[stump_idx : stump_end + 1]))
            stump_idx += d

    if stump_idx + 1 <= len(background_stumps):
        left_stumps = background_stumps[stump_idx - 1 :]
        # note that these size-1 hyperedges might be duplicate with he1 generated earlier
        if params.q[0] > 0 and (
            not params.is_simple
            or (len(left_stumps) == len(np.unique(left_stumps)))
            and len(np.intersect1d([he1_[0] for he1_ in he1], left_stumps)) == 0
        ):
            edges.extend([[x] for x in left_stumps])
        else:
            find_first = 0
            find_some = [i for i in range(len(params.q[1:])) if params.q[i] > 0]
            if find_some:
                find_first = find_some[0]
            targetq = 1 + find_first
            to_add = targetq - len(left_stumps)
            assert to_add > 0
            extra_stumps = random.choices(
                range(len(params.w)), weights=params.w, k=to_add
            )
            left_stumps.append(*extra_stumps)
            assert len(left_stumps) == targetq
            edges.append(sorted(left_stumps))
            print(
                f"added degree to the following nodes due to parity issues: {extra_stumps}"
            )

    edges.extend(he1)
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    all_sorted = [is_sorted(e) for e in edges]
    assert np.all(all_sorted)

    if params.is_simple:
        bad_multi = 0
        bad_dup = 0
        good_edges = set[tuple[int]]()
        bad_edges = list[list[int]]()
        for he in edges:
            not_multi = len(he) == len(np.unique(he))
            not_dup = tuple(he) not in good_edges
            bad_multi += not not_multi
            bad_dup += not not_dup
            if not_multi and not_dup:
                good_edges.add(tuple(he))
            else:
                bad_edges.append(he)

        if bad_multi > 0:
            bad_multi_pct = round(100 * bad_multi / len(edges), ndigits=2)
            print(
                f"fixing {bad_multi} hyperedges ({bad_multi_pct}% of total number of hyperedges) that were multisets"
            )
        if bad_dup > 0:
            bad_dup_pct = round(100 * bad_dup / len(edges), ndigits=2)
            print(
                f"fixing {bad_dup} hyperedges ({bad_dup_pct}% of total number of hyperedges) that were duplicated"
            )

        random.shuffle(bad_edges)

        iters = params.maxiter * len(bad_edges)
        for _ in range(iters):
            if not bad_edges:
                break

            bad_edge = bad_edges.pop()
            initial_badness = (tuple(bad_edge) in good_edges) + (
                len(bad_edge) - len(np.unique(bad_edge))
            )
            if initial_badness == 0:
                good_edges.add(tuple(bad_edge))
            else:
                other_edge = random.choice(list(good_edges))
                good_edges.remove(other_edge)
                new_split: list[int] = [*bad_edge, *other_edge]
                random.shuffle(new_split)
                new1 = tuple(sorted(new_split[: len(bad_edge)]))
                new2 = tuple(sorted(new_split[len(bad_edge) :]))
                final_bandess = (
                    (new1 in good_edges)
                    + (len(new1) - len(np.unique(new1)))
                    + (new2 in good_edges)
                    + (len(new2) - len(np.unique(new2)))
                )
                if final_bandess < initial_badness:
                    if len(new1) == len(np.unique(new1)) and new1 not in good_edges:
                        good_edges.add(new1)
                    else:
                        bad_edges.append(list(new1))

                    if len(new2) == len(np.unique(new2)) and new2 not in good_edges:
                        good_edges.add(new2)
                    else:
                        bad_edges.append(list(new2))

                else:
                    bad_edges.insert(0, bad_edge)
                    good_edges.add(other_edge)

        if bad_edges:
            print(
                f"""Failed to fix all bad edges in {params.maxiter} rounds.
                  Dropping {len(bad_edges)} bad edges that violated
                  simple hypergraph condition."""
            )
            print(bad_edges)

        edges = [list(g_e) for g_e in good_edges]

    all_sorted2 = [is_sorted(e) for e in edges]
    assert np.all(all_sorted2)

    return edges


def trunc_powerlaw_weigths(a, v_min, v_max):
    assert a >= 1
    assert 1 <= v_min <= v_max
    return [1 / i**a for i in range(v_min, v_max + 1)]


def sample_trunc_powerlaw(a, v_min, v_max, n):
    if isinstance(a, list):
        assert n > 0
        return random.choices(range(v_min, v_max + 1), k=n, weights=a)

    assert a >= 1
    assert 1 <= v_min <= v_max
    assert n > 0
    w = [1 / i**a for i in range(v_min, v_max + 1)]
    return random.choices(range(v_min, v_max + 1), k=n, weights=w)


def sample_degrees(t1, d_min, d_max, n):
    """
        sample_degrees(τ₁, d_min, d_max, n)

    Return a vector of length `n` of sampled degrees of vertices following a truncated
    discrete power law distribution with truncation range `[d_min, d_max]` and exponent `τ₁`.

    The producedure does not check if the returned vector is a graphical degree sequence.
    """
    s = sample_trunc_powerlaw(t1, d_min, d_max, n)
    return sorted(s, reverse=True)


def sample_communities(t2, c_min, c_max, n, max_iter):
    """
        sample_communities(τ₂, c_min, c_max, n, max_iter)

    Return a vector of sampled community sizes following a truncated
    discrete power law distribution with truncation range `[c_min, c_max]` and exponent `τ₂`.
    The sum of sizes is equal to number of vertices in the graph `n`.

    The sampling is attempted `max_iter` times to find an admissible result.
    In case of failure a correction of community sizes is applied to the sampled sequence
    that was closest to a feasible one to ensure that the result is admissible.
    """
    assert 1 <= c_min <= c_max
    l_min = n / c_max
    l_max = n / c_min
    assert l_min >= 1
    assert ceil(l_min) <= floor(l_max)

    best_s = []
    best_ss = np.iinfo(int).max

    w = trunc_powerlaw_weigths(t2, c_min, c_max)
    for i in range(max_iter):
        s = sample_trunc_powerlaw(w, c_min, c_max, ceil(l_max))
        stopidx = -1
        ss = 0
        while ss < n:
            stopidx += 1
            ss += s[stopidx]

        if ss == n:
            return sorted(s[: stopidx + 1], reverse=True)

        if ss < best_ss:
            best_ss = ss
            best_s = s[: stopidx + 1]

    print(
        f"failed to sample an admissible community sequence in {max_iter} draws. Fixing"
    )

    random.shuffle(best_s)

    if len(best_s) > l_max:
        best_s = best_s[: l_max + 1]
        best_ss = sum(best_s)

    i = -1
    while best_ss != n:
        if i >= len(best_s) - 1:
            i = -1
            random.shuffle(best_s)

        i += 1
        change = copysign(1, n - best_ss)

        if change > 0:
            if not best_s[i] < c_max:
                continue
        else:
            if not best_s[i] > c_min:
                continue

        best_ss += change
        best_s[i] += change
    return sorted(best_s, reverse=True)


def gen_hypergraph(params):
    """
        gen_hypergraph(params:ABCDHParams)

    Generate ABCD hypergraph following parameters specified in `params`.

    Return a named tuple containing a vector of hyperedges of the graph and a list of cluster
    assignments of the vertices.
    The ordering of vertices and clusters is in descending order (as in `params`).
    """
    he1: list[list[int]] = generate_1hyperedges(params)

    for i, _ in enumerate(params.w):
        params.z[i] = randround(params.x * params.w[i])
    params.y = [params.w[i] - params.z[i] for i in range(len(params.w))]

    clusters = populate_clusters(params)
    hyperedges = config_model(clusters, params, he1)

    return hyperedges, clusters


def abcdh_hypergraph(n, dss, css, x, q, ws, seed, m=False):
    if not np.issubdtype(type(seed), np.integer):
        warnings.warn("seed must be an integer")
    random.seed(seed)

    if not np.issubdtype(type(n), np.integer):
        warnings.warn("Number of vertices must be an integer")
    if n <= 0:
        warnings.warn("Number of vertices must be positive")

    if len(dss) == 3:
        g = dss[0]
        if not np.issubdtype(type(g), np.float64):
            warnings.warn("g must be a number")
        if not 2 < g < 3:
            warnings.warn("g must be in (2, 3)")
        d = dss[1]
        if not np.issubdtype(type(d), np.integer):
            warnings.warn("Number of vertices must be an integer")
        D = dss[2]
        if not np.issubdtype(type(D), np.integer):
            warnings.warn("Number of vertices must be an integer")
        if not 0 < d <= D:
            warnings.warn("Condition 0 < d <= D not met")

        degs: list = sample_degrees(g, d, D, n)

    if len(css) == 3:
        b = css[0]
        if not np.issubdtype(type(b), np.float64):
            warnings.warn("b must be a number")
        if not 1 < b < 2:
            warnings.warn("b must be in (1, 2)")
        s = css[1]
        if not np.issubdtype(type(s), np.integer):
            warnings.warn("Number of vertices must be an integer")
        S = css[2]
        if not np.issubdtype(type(S), np.integer):
            warnings.warn("Number of vertices must be an integer")
        if not d <= s <= S:
            warnings.warn("Condition d <= s <= S not met")

        coms: list = sample_communities(b, s, S, n, 1000)

    if n != sum(coms):
        warnings.warn(
            f"number of vertices {n} does not match the sum of community sizes {sum(coms)}"
        )

    if not np.issubdtype(type(x), np.float64):
        warnings.warn("x must be a number")
    if not 0 <= x <= 1:
        warnings.warn("x must be in [0, 1]")

    w = np.zeros((len(q), len(q)), dtype=float)

    if ws == ":strict":
        for d in range(len(q)):
            w[d, d] = 1
    elif ws == ":linear":
        for d in range(len(q)):
            for c in range((d + 1) // 2, d + 1):
                w[c, d] = c + 1
            w[:, d] = [i / sum(w[:, d]) for i in w[:, d]]
    elif ws == ":majority":
        for d in range(len(q)):
            for c in range((d + 1) // 2, d + 1):
                w[c, d] = 1.0 / (d + 1 - (d + 1) // 2)

    params = ABCDH_params(degs, coms, x, q, w, not m, 100)
    hyperedges, clusters = gen_hypergraph(params)

    return hyperedges, clusters


if __name__ == "__main__":
    he, cl = abcdh_hypergraph(
        10000,
        [2.5, 5, 100],
        [1.5, 100, 1000],
        0.5,
        [0.0, 0.4, 0.3, 0.2, 0.1],
        ":linear",
        1234,
    )
    print("hyperedges\n", he, "\n\n")
    print("clusters\n", cl, "\n\n")

    print(len(he))

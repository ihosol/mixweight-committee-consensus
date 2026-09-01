package keeper

import (
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// riskBudgetKnobs holds the configuration of the risk-budget controller path.
// All knobs are loaded from environment variables in v1 (no protobuf changes).
type riskBudgetKnobs struct {
	Enabled        bool
	Mode           string  // "signal" | "risk" | "hybrid"
	Epsilon        float64 // policy budget for committee-capture certificate
	Theta          float64 // capture threshold (e.g., 1/3 for BFT safety)
	CoalitionMode  string  // "min_share" | "top_k"
	CoalitionShare float64 // min_share: smallest top-stake prefix reaching this mass
	CoalitionTopK  int     // top_k: fixed number of top-stake validators
	GridStep       float64 // grid search step over [0, lambdaMax]
}

// defaultRiskBudgetKnobs reads the risk-budget knobs from environment variables
// using the same pattern as defaultAdaptiveControllerKnobs. Defaults match
// papers/risk_budget_controller_2026/POC_RESULTS_SPEC.md §1.
func defaultRiskBudgetKnobs() riskBudgetKnobs {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("ADAPTIVE_CONTROLLER_MODE")))
	if mode == "" {
		mode = "signal"
	}
	enabled := envBool("RISK_BUDGET_ENABLED", false)
	if mode == "hybrid" || mode == "risk" {
		enabled = true
	}
	coalitionMode := strings.ToLower(strings.TrimSpace(os.Getenv("RISK_BUDGET_COALITION_MODE")))
	if coalitionMode != "top_k" {
		coalitionMode = "min_share"
	}
	return riskBudgetKnobs{
		Enabled:        enabled,
		Mode:           mode,
		Epsilon:        envFloatAny("RISK_BUDGET_EPS", 1.0e-4),
		Theta:          float64(envUint64("RISK_BUDGET_THETA_PPM", 333333)) / 1_000_000.0,
		CoalitionMode:  coalitionMode,
		CoalitionShare: float64(envUint64("RISK_BUDGET_COALITION_SHARE_PPM", 333333)) / 1_000_000.0,
		CoalitionTopK:  int(envUint64("RISK_BUDGET_TOP_K", 3)),
		GridStep:       float64(envUint64("RISK_BUDGET_GRID_STEP_PPM", 10000)) / 1_000_000.0,
	}
}

// selectCoalitionTopK returns the indices of the fixed top-k validators by
// stake (descending). Unlike selectCoalition (min_share), the coalition size
// is constant across epochs, which keeps alpha/beta trajectories interpretable
// under concentration drift.
func selectCoalitionTopK(stakes []uint64, k int) []int {
	n := len(stakes)
	if n == 0 || k <= 0 {
		return nil
	}
	if k > n {
		k = n
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return stakes[order[a]] > stakes[order[b]]
	})
	return order[:k]
}

// envBool parses a 0/1 environment variable; defaults to fallback when unset
// or unparsable.
func envBool(name string, fallback bool) bool {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return fallback
	}
	n, err := strconv.ParseUint(v, 10, 64)
	if err != nil {
		return fallback
	}
	return n != 0
}

// envFloatAny parses an arbitrary positive float (scientific notation accepted)
// — used for risk-budget epsilon values that fall outside the [0,1] range
// envFloat01 enforces.
func envFloatAny(name string, fallback float64) float64 {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || f <= 0 {
		return fallback
	}
	return f
}

// baselineModeEnv reads the BASELINE_MODE knob that selects which corrective
// baseline u_i the mixed-weight rule and the risk-budget certificate use.
//
//	service_age  (default) u_i ∝ ageT_i · stake_i  — persistence path,
//	                       appropriate for anti-Sybil / fresh-entry regimes
//	                       (Article 2 lineage). Beta tracks stake when ages are
//	                       homogeneous, so it does NOT deconcentrate mature stake.
//	capped_stake          u_i ∝ min(stake_i, cap)  — concentration path,
//	                       appropriate for whale-drift regimes. A whale that
//	                       accumulates past the cap stops gaining baseline mass,
//	                       so beta grows slower than alpha (correct anti-
//	                       concentration sign). cap = capPpm · totalStake.
//	uniform               u_i ∝ 1/N               — Article 1 lineage; the
//	                       degenerate concentration baseline.
func baselineModeEnv() string {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("BASELINE_MODE")))
	switch v {
	case "capped_stake", "concave_stake", "uniform", "service_age":
		return v
	default:
		return "service_age"
	}
}

// buildBaselineWeights returns per-validator baseline weights and their sum for
// the active baseline mode. serviceAge is the precomputed ageT_i·stake_i slice
// already used by the persistence path; it is ignored for the other modes.
// capPpm is interpreted as a fraction of totalStake in ppm and only used by
// capped_stake. The returned weights are unnormalized; callers normalize via
// ppmRatio(weight, total, scale).
func buildBaselineWeights(mode string, stakes, serviceAge []uint64, totalStake, capPpm uint64) ([]uint64, uint64) {
	n := len(stakes)
	out := make([]uint64, n)
	var total uint64
	switch mode {
	case "uniform":
		for i := range out {
			out[i] = 1
		}
		total = uint64(n)
	case "capped_stake":
		// cap in absolute stake tokens; guard against zero/overflow.
		cap := uint64(0)
		if capPpm > 0 && totalStake > 0 {
			cap = uint64((float64(totalStake) * float64(capPpm)) / 1_000_000.0)
		}
		for i, s := range stakes {
			v := s
			if cap > 0 && v > cap {
				v = cap
			}
			if v == 0 {
				v = 1
			}
			out[i] = v
			total += v
		}
	case "concave_stake":
		// Anti-concentration baseline u_i ∝ stake_i^γ, γ∈(0,1] (γ=0.5 is Square
		// Root Stake Weight). A smooth sublinear transform: it compresses large
		// stakes so beta grows slower than alpha under drift (correct anti-
		// concentration sign), without the hard-cap kink. γ is read from
		// RISK_BUDGET_CONCAVE_GAMMA_PPM (ppm; default 500000 = 0.5).
		// NOTE: math.Pow is acceptable for the single-host PoC localnet
		// (deterministic across identical binaries); a production deployment must
		// pin γ=0.5 via math.Sqrt or a fixed-point power for cross-platform
		// consensus determinism.
		gamma := float64(envUint64("RISK_BUDGET_CONCAVE_GAMMA_PPM", 500000)) / 1_000_000.0
		for i, s := range stakes {
			if s == 0 {
				out[i] = 1
				total += 1
				continue
			}
			w := uint64(math.Pow(float64(s), gamma)*1_000.0 + 0.5)
			if w == 0 {
				w = 1
			}
			out[i] = w
			total += w
		}
	default: // service_age
		for i := range serviceAge {
			out[i] = serviceAge[i]
			total += serviceAge[i]
		}
	}
	if total == 0 {
		total = 1
	}
	return out, total
}

// selectCoalition returns the smallest top-stake prefix (as indices into the
// input stakes slice) whose cumulative stake share reaches phi. The smallest
// returned coalition has at least one validator (the top-stake one) when phi>0
// and totalStake>0.
func selectCoalition(stakes []uint64, totalStake uint64, phi float64) []int {
	n := len(stakes)
	if n == 0 || totalStake == 0 || phi <= 0 {
		return nil
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return stakes[order[a]] > stakes[order[b]]
	})
	target := float64(totalStake) * phi
	var cumulative float64
	coalition := make([]int, 0, n)
	for _, idx := range order {
		coalition = append(coalition, idx)
		cumulative += float64(stakes[idx])
		if cumulative >= target {
			return coalition
		}
	}
	return coalition
}

// coalitionMasses returns the stake mass alpha and the baseline mass beta for
// the monitored coalition. The `ages` slice carries the service-age proxy
// values already used by msg_server_draw_committee.go (age_transform * stake),
// not raw block ages. The returned masses are normalized shares in [0,1].
func coalitionMasses(coalition []int, stakes, ages []uint64, totalStake, totalAge uint64) (alpha, beta float64) {
	if len(coalition) == 0 || totalStake == 0 || totalAge == 0 {
		return 0, 0
	}
	var sumStake, sumAge uint64
	for _, idx := range coalition {
		if idx < 0 || idx >= len(stakes) || idx >= len(ages) {
			continue
		}
		sumStake += stakes[idx]
		sumAge += ages[idx]
	}
	alpha = float64(sumStake) / float64(totalStake)
	beta = float64(sumAge) / float64(totalAge)
	return alpha, beta
}

// binomialUpperTail returns P(X >= q) for X ~ Binomial(m, p), by exact
// summation. The committee sizes used by the PoC are small, so the direct
// float64 sum is accurate enough; the PMF is advanced iteratively to avoid
// factorial overflow. Same single-host PoC determinism caveat as math.Pow
// elsewhere in this file.
func binomialUpperTail(p float64, m, q uint64) float64 {
	if q == 0 {
		return 1
	}
	if q > m {
		return 0
	}
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return 1
	}

	pmf := math.Pow(1-p, float64(m))
	ratio := p / (1 - p)
	var tail float64
	for i := uint64(0); i <= m; i++ {
		if i >= q {
			tail += pmf
		}
		if i == m {
			break
		}
		pmf = pmf * float64(m-i) / float64(i+1) * ratio
	}
	if tail > 1 {
		return 1
	}
	if tail < 0 {
		return 0
	}
	return tail
}

// binomialCaptureTail returns B_t(lambda): the exact-binomial upper bound on
// the probability that the monitored coalition takes at least ceil(theta*m) of
// the m committee seats, treating each seat as an independent Bernoulli draw
// with coalition selection mass p(lambda) = (1-lambda)*alpha + lambda*beta.
// By Hoeffding (1963) this upper-bounds the true without-replacement capture,
// so the guarantee stays conservative while being far tighter than the
// Chernoff–KL bound it replaces. Returns a value in [0,1].
func binomialCaptureTail(alpha, beta float64, m uint64, theta, lambda float64) float64 {
	if m == 0 || theta <= 0 || theta > 1 {
		return 1
	}
	p := (1-lambda)*alpha + lambda*beta
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	q := uint64(math.Ceil(theta * float64(m)))
	if q == 0 {
		q = 1
	}
	return binomialUpperTail(p, m, q)
}

// riskBudgetTarget returns the smallest lambda in [0, lambdaMax] (at the given
// gridStep) such that B_t(lambda) <= epsilon. The second return value is true
// when the budget is satisfied at the returned lambda.
//
// When no grid point satisfies the budget, the function returns the lambda that
// MINIMIZES B_t over the grid (best-effort intervention) together with
// satisfied=false. This avoids the failure mode where raising lambda makes
// things worse — which is possible when the baseline mass for the monitored
// coalition exceeds its stake mass (beta > alpha), making B_t monotone
// increasing in lambda. In that regime the smallest available B_t is at
// lambda=0, and returning lambda=0 is strictly safer than returning lambdaMax.
func riskBudgetTarget(alpha, beta float64, m uint64, theta, epsilon, lambdaMax, gridStep float64) (float64, bool) {
	if epsilon <= 0 || epsilon >= 1 || lambdaMax <= 0 {
		return 0, false
	}
	if gridStep <= 0 {
		gridStep = 0.01
	}
	bestLambda := 0.0
	bestBound := binomialCaptureTail(alpha, beta, m, theta, 0)
	if bestBound <= epsilon {
		return 0, true
	}
	for lambda := gridStep; lambda < lambdaMax+gridStep/2; lambda += gridStep {
		lam := lambda
		if lam > lambdaMax {
			lam = lambdaMax
		}
		b := binomialCaptureTail(alpha, beta, m, theta, lam)
		if b <= epsilon {
			return lam, true
		}
		if b < bestBound {
			bestBound = b
			bestLambda = lam
		}
		if lam >= lambdaMax {
			break
		}
	}
	return bestLambda, false
}

// log10ToFixedPoint encodes a positive bound value as round(1e6 * log10(value)).
// Returns a sentinel (math.MinInt64) when the value is non-positive so callers
// can flag "below resolution" in diagnostics.
func log10ToFixedPoint(value float64) int64 {
	if value <= 0 {
		return math.MinInt64
	}
	return int64(math.Round(1.0e6 * math.Log10(value)))
}

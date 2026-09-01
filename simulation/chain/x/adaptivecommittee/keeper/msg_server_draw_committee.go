package keeper

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/big"
	"math/bits"
	"os"
	"sort"
	"strconv"
	"strings"

	"chain-five-three/x/adaptivecommittee/types"

	sdk "github.com/cosmos/cosmos-sdk/types"
)

func checkedMul(a, b uint64) (uint64, bool) {
	hi, lo := bits.Mul64(a, b)
	if hi != 0 {
		return 0, false
	}
	return lo, true
}

func checkedAdd(a, b uint64) (uint64, bool) {
	s, c := bits.Add64(a, b, 0)
	if c != 0 {
		return 0, false
	}
	return s, true
}

func ppmRatio(num, den, scale uint64) uint64 {
	if den == 0 {
		return 0
	}
	n := new(big.Int).SetUint64(num)
	n.Mul(n, new(big.Int).SetUint64(scale))
	d := new(big.Int).SetUint64(den)
	n.Div(n, d)
	if !n.IsUint64() {
		return scale
	}
	v := n.Uint64()
	if v > scale {
		return scale
	}
	return v
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// drawPolicyFromTag provides an experiment-safe policy switch without protobuf changes.
// Tag conventions:
//   manual__<rest>   -> use manual lambda from state (set via set-lambda tx)
//   adaptive__<rest> -> force adaptive auto-controller path
//   default (no prefix) keeps backward compatibility and behaves as adaptive.
func drawPolicyFromTag(tag string) string {
	if strings.HasPrefix(tag, "manual__") {
		return "manual"
	}
	if strings.HasPrefix(tag, "adaptive__") {
		return "adaptive"
	}
	return "adaptive"
}

func giniFromStakes(stakes []uint64) float64 {
	n := len(stakes)
	if n == 0 {
		return 0
	}
	x := make([]float64, 0, n)
	var sum float64
	for _, s := range stakes {
		v := float64(s)
		x = append(x, v)
		sum += v
	}
	if sum <= 0 {
		return 0
	}
	sort.Float64s(x)
	var weighted float64
	for i, v := range x {
		weighted += float64(i+1) * v
	}
	g := (2.0*weighted)/(float64(n)*sum) - (float64(n)+1.0)/float64(n)
	return clamp01(g)
}

func freshPressureFromAge(stakes []uint64, agesRaw []uint64, totalStake uint64, horizonBlocks float64) float64 {
	if len(stakes) == 0 || len(stakes) != len(agesRaw) || totalStake == 0 {
		return 0
	}
	if horizonBlocks <= 0 {
		horizonBlocks = 40.0
	}

	// Weighted freshness dispersion: low at startup / homogeneous age, high when fresh entrants
	// appear against an older incumbent set.
	den := float64(totalStake)
	freshVals := make([]float64, len(stakes))
	var mean float64
	for i := range stakes {
		w := float64(stakes[i]) / den
		f := math.Exp(-float64(agesRaw[i]) / horizonBlocks)
		freshVals[i] = f
		mean += w * f
	}

	var varW float64
	for i := range stakes {
		w := float64(stakes[i]) / den
		d := freshVals[i] - mean
		varW += w * d * d
	}

	stdW := math.Sqrt(varW)
	return clamp01(stdW)
}

type adaptiveControllerKnobs struct {
	LamMax              float64
	AlphaUp             float64
	AlphaDown           float64
	HysteresisFloor     float64
	HysteresisTrigger   float64
	FreshnessWeight     float64
	GiniWeight          float64
	SplitWeight         float64
	FreshnessNormOffset float64
	FreshnessNormSpan   float64
	GiniNormOffset      float64
	GiniNormSpan        float64
}

func envFloat01(name string, fallback float64) float64 {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return fallback
	}
	return clamp01(f)
}

func envUint64(name string, fallback uint64) uint64 {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return fallback
	}
	n, err := strconv.ParseUint(v, 10, 64)
	if err != nil || n == 0 {
		return fallback
	}
	return n
}

func trackedValidatorPrefix() string {
	v := strings.TrimSpace(os.Getenv("TRACKED_VALIDATOR_PREFIX"))
	if v == "" {
		return "sybil"
	}
	return v
}

// committeeDrawMode selects the committee sampling scheme via the
// COMMITTEE_DRAW_MODE environment variable:
//   "wr"  -> with replacement (i.i.d. multi-seat sortition; coalition seat count
//            is exactly Binomial(m, p_t(lambda)), so the binomial certificate is
//            the exact law of the draw). Used by the risk-budget assessment.
//   "wor" -> without replacement (distinct committee members; the binomial is then
//            a conservative certificate). Used by the adaptive-defence experiments.
// Defaults to "wor" for backward compatibility with the published defence package.
func committeeDrawMode() string {
	if strings.ToLower(strings.TrimSpace(os.Getenv("COMMITTEE_DRAW_MODE"))) == "wr" {
		return "wr"
	}
	return "wor"
}

func defaultAdaptiveControllerKnobs(k Keeper, ctx sdk.Context) adaptiveControllerKnobs {
	s := float64(LambdaScalePpm())
	knobs := adaptiveControllerKnobs{
		LamMax:              float64(k.GetAdaptiveLamMaxPpm(ctx)) / s,
		AlphaUp:             float64(k.GetAdaptiveAlphaUpPpm(ctx)) / s,
		AlphaDown:           float64(k.GetAdaptiveAlphaDownPpm(ctx)) / s,
		HysteresisFloor:     float64(k.GetAdaptiveHysteresisFloorPpm(ctx)) / s,
		HysteresisTrigger:   float64(k.GetAdaptiveHysteresisTriggerPpm(ctx)) / s,
		FreshnessWeight:     float64(k.GetAdaptiveFreshnessWeightPpm(ctx)) / s,
		GiniWeight:          float64(k.GetAdaptiveGiniWeightPpm(ctx)) / s,
		SplitWeight:         float64(k.GetAdaptiveSplitWeightPpm(ctx)) / s,
		FreshnessNormOffset: float64(k.GetAdaptiveFreshnessNormOffsetPpm(ctx)) / s,
		FreshnessNormSpan:   float64(k.GetAdaptiveFreshnessNormSpanPpm(ctx)) / s,
		GiniNormOffset:      float64(k.GetAdaptiveGiniNormOffsetPpm(ctx)) / s,
		GiniNormSpan:        float64(k.GetAdaptiveGiniNormSpanPpm(ctx)) / s,
	}
	knobs.LamMax = envFloat01("ADAPTIVE_LAM_MAX", knobs.LamMax)
	knobs.AlphaUp = envFloat01("ADAPTIVE_ALPHA_UP", knobs.AlphaUp)
	knobs.AlphaDown = envFloat01("ADAPTIVE_ALPHA_DOWN", knobs.AlphaDown)
	knobs.HysteresisFloor = envFloat01("ADAPTIVE_HYST_FLOOR", knobs.HysteresisFloor)
	knobs.HysteresisTrigger = envFloat01("ADAPTIVE_HYST_TRIGGER", knobs.HysteresisTrigger)
	knobs.FreshnessWeight = envFloat01("ADAPTIVE_FRESHNESS_W", knobs.FreshnessWeight)
	knobs.GiniWeight = envFloat01("ADAPTIVE_GINI_W", knobs.GiniWeight)
	knobs.SplitWeight = envFloat01("ADAPTIVE_SPLIT_W", knobs.SplitWeight)
	knobs.FreshnessNormOffset = envFloat01("ADAPTIVE_FRESHNESS_NORM_F0", knobs.FreshnessNormOffset)
	knobs.FreshnessNormSpan = envFloat01("ADAPTIVE_FRESHNESS_NORM_FSPAN", knobs.FreshnessNormSpan)
	knobs.GiniNormOffset = envFloat01("ADAPTIVE_GINI_NORM_G0", knobs.GiniNormOffset)
	knobs.GiniNormSpan = envFloat01("ADAPTIVE_GINI_NORM_GSPAN", knobs.GiniNormSpan)
	if knobs.FreshnessNormSpan <= 0 {
		knobs.FreshnessNormSpan = float64(defaultAdaptiveFreshnessNormSpanPpm) / s
	}
	if knobs.GiniNormSpan <= 0 {
		knobs.GiniNormSpan = float64(defaultAdaptiveGiniNormSpanPpm) / s
	}
	return knobs
}

func adaptiveScore(fNorm, gNorm, splitPressure float64, knobs adaptiveControllerKnobs) float64 {
	return clamp01(knobs.FreshnessWeight*fNorm + knobs.GiniWeight*gNorm + knobs.SplitWeight*splitPressure)
}

// adaptiveSignalTarget returns the pre-filter mixing target derived from the
// composite signal score: lambda^sig = score * LamMax.
func adaptiveSignalTarget(score float64, knobs adaptiveControllerKnobs) float64 {
	return score * knobs.LamMax
}

// adaptiveLambdaShell applies the asymmetric first-order filter and the
// hysteresis floor to a pre-filter target. The signal-score-driven hysteresis
// only fires when riskOnly=false; in pure risk mode the signal path is
// excluded by construction, so the signal score must NOT contaminate the
// realized lambda through HysteresisFloor. In signal and hybrid modes the
// hysteresis behavior follows the Article 2 semantics: a sufficiently large
// signal score holds lambda above the floor for one update step.
func adaptiveLambdaShell(lamPrev, lamTarget, score float64, knobs adaptiveControllerKnobs, riskOnly bool) float64 {
	alpha := knobs.AlphaDown
	if lamTarget > lamPrev {
		alpha = knobs.AlphaUp
	}
	lamAuto := (1.0-alpha)*lamPrev + alpha*lamTarget
	if !riskOnly && score >= knobs.HysteresisTrigger {
		lamAuto = math.Max(lamAuto, knobs.HysteresisFloor)
	}
	if score == 0 && lamPrev < 0.001 && lamTarget == 0 {
		lamAuto = 0
	}
	return clamp01(lamAuto)
}

// adaptiveLambdaNext is the signal-only path used when the risk-budget path is
// disabled. Kept as a backward-compatible wrapper over the shell.
func adaptiveLambdaNext(lamPrev, score float64, knobs adaptiveControllerKnobs) float64 {
	return adaptiveLambdaShell(lamPrev, adaptiveSignalTarget(score, knobs), score, knobs, false)
}

func mixedWeightPpm(stakePpm, basePpm, lam, scale uint64) (uint64, error) {
	term1, ok := checkedMul(scale-lam, stakePpm)
	if !ok {
		return 0, fmt.Errorf("weight overflow (mix term1)")
	}
	term2, ok := checkedMul(lam, basePpm)
	if !ok {
		return 0, fmt.Errorf("weight overflow (mix term2)")
	}
	w, ok := checkedAdd(term1, term2)
	if !ok {
		return 0, fmt.Errorf("weight overflow (mix add)")
	}
	if w == 0 {
		w = 1
	}
	return w, nil
}

func (k msgServer) DrawCommittee(goCtx context.Context, msg *types.MsgDrawCommittee) (*types.MsgDrawCommitteeResponse, error) {
	ctx := sdk.UnwrapSDKContext(goCtx)

	if msg.Size_ == 0 {
		return nil, fmt.Errorf("committee size must be positive")
	}

	// Fetch validator set (bonded, power-sorted).
	vals, err := k.stakingKeeper.GetBondedValidatorsByPower(goCtx)
	if err != nil {
		return nil, err
	}
	if len(vals) == 0 {
		return nil, fmt.Errorf("no bonded validators")
	}

	// Implementation of Persistence-Weighted Committee Selection (Sybil Defense).
	// Mixes stake-proportional weight (p_i) with persistence/age weight (u_i).
	// Formula: q_i(λ) = (1-λ) * p_i + λ * u_i
	// where u_i = age_i / sum(age)

	lamManual := k.GetLambdaPpm(ctx)
	lam := lamManual
	S := LambdaScalePpm()
	policyMode := drawPolicyFromTag(msg.Tag)

	// Collect stake and compute TotalStake
	// Collect creation heights to compute Age/Persistence

	N := len(vals)
	stakes := make([]uint64, N)
	addrs := make([]string, N)
	ages := make([]uint64, N)
	agesRaw := make([]uint64, N)
	isAttacker := make([]bool, N)
	var totalStake uint64
	var totalAge uint64
	var attackerStake uint64
	var attackerAge uint64
	var attackerCount int

	curH := uint64(1)
	if ctx.BlockHeight() > 0 {
		curH = uint64(ctx.BlockHeight())
	}

	tauMax := k.GetPersistenceTauMaxBlocks(ctx)
	zetaPpm := k.GetPersistenceZetaPpm(ctx)

	trackedPrefix := trackedValidatorPrefix()
	for i, v := range vals {
		// 1. Stake
		t := v.GetTokens()
		if !t.IsUint64() {
			return nil, fmt.Errorf("validator tokens overflow uint64")
		}
		s := t.Uint64()
		stakes[i] = s
		var okAdd bool
		totalStake, okAdd = checkedAdd(totalStake, s)
		if !okAdd {
			return nil, fmt.Errorf("total stake overflow")
		}
		op := v.GetOperator()
		addrs[i] = op
		mon := v.Description.Moniker
		if strings.HasPrefix(mon, trackedPrefix) {
			isAttacker[i] = true
			attackerStake, okAdd = checkedAdd(attackerStake, s)
			if !okAdd {
				return nil, fmt.Errorf("attacker stake overflow")
			}
			attackerCount++
		}

		// 2. Persistence age from module-owned first-seen height.
		firstSeen, ok := k.GetValidatorFirstSeenHeight(ctx, op)
		if !ok {
			// If not seen yet, initialize to current height.
			firstSeen = curH
			k.SetValidatorFirstSeenHeight(ctx, op, firstSeen)
		}

		var age uint64 = 1
		if curH >= firstSeen {
			age = (curH - firstSeen) + 1
		}
		agesRaw[i] = age

		ageT := k.ApplyPersistenceTransform(age, tauMax, zetaPpm)
		if ageT == 0 {
			ageT = 1
		}

		// Service-age proxy (stake-age): expected validated work ~= age * stake.
		// This penalizes fresh split sybils more strongly than plain validator-age.
		serviceAge, okMul := checkedMul(ageT, s)
		if !okMul {
			return nil, fmt.Errorf("service-age overflow")
		}
		if serviceAge == 0 {
			serviceAge = 1
		}

		ages[i] = serviceAge
		totalAge, okAdd = checkedAdd(totalAge, serviceAge)
		if !okAdd {
			return nil, fmt.Errorf("total age overflow")
		}
		if isAttacker[i] {
			attackerAge, okAdd = checkedAdd(attackerAge, serviceAge)
			if !okAdd {
				return nil, fmt.Errorf("attacker age overflow")
			}
		}
	}

	// Diagnostics are always computed/emitted for observability, even in manual mode.
	// Controller knobs include the affine normalizer offsets/spans used for the
	// freshness-dispersion and stake-Gini signals, so we load them unconditionally
	// and reuse the same struct in the adaptive branch below.
	horizonBlocks := envUint64("ADAPTIVE_FRESHNESS_HORIZON_BLOCKS", k.GetAdaptiveFreshnessHorizonBlocks(ctx))
	knobs := defaultAdaptiveControllerKnobs(k.Keeper, ctx)
	freshPressure := freshPressureFromAge(stakes, agesRaw, totalStake, float64(horizonBlocks))
	gini := giniFromStakes(stakes)
	fNorm := clamp01((freshPressure - knobs.FreshnessNormOffset) / knobs.FreshnessNormSpan)
	gNorm := clamp01((gini - knobs.GiniNormOffset) / knobs.GiniNormSpan)

	// Sybil-blind split-pressure signal: detect when validators classified as "fresh"
	// by age threshold hold disproportionately less stake than their count share.
	// This replaces the earlier label-based signal (attackerValShare - attackerStakeShare)
	// which inappropriately let the controller see ground-truth attacker labels.
	freshCountCtrl := 0
	var freshStakeCtrl uint64
	for i := range vals {
		if agesRaw[i] < horizonBlocks {
			freshCountCtrl++
			freshStakeCtrl, _ = checkedAdd(freshStakeCtrl, stakes[i])
		}
	}
	freshCountShare := 0.0
	if N > 0 {
		freshCountShare = float64(freshCountCtrl) / float64(N)
	}
	freshStakeShare := 0.0
	if totalStake > 0 {
		freshStakeShare = float64(freshStakeCtrl) / float64(totalStake)
	}
	splitPressure := clamp01(freshCountShare - freshStakeShare)

	// Dual-path baseline selection (RBHC paper). The active baseline u_i drives
	// BOTH the mixed-weight rule and the risk-budget certificate, so alpha/beta
	// and the realized committee weights stay mutually consistent.
	//   service_age  -> persistence path  (anti-Sybil; ages[] = ageT*stake)
	//   capped_stake -> concentration path (anti-whale; min(stake, cap))
	//   uniform      -> Article 1 degenerate baseline
	baselineMode := baselineModeEnv()
	concentrationCapPpm := envUint64("CONCENTRATION_CAP_PPM", 125000)
	baseWeights, totalBase := buildBaselineWeights(baselineMode, stakes, ages, totalStake, concentrationCapPpm)

	// Risk-budget diagnostics are computed regardless of controller mode so
	// every draw event records the policy-facing certificate B_t(0) and
	// B_t(lambda_auto). This lets the harness compare controllers fairly even
	// in the signal-only baseline. beta is taken from the ACTIVE baseline.
	rbKnobs := defaultRiskBudgetKnobs()
	var coalition []int
	if rbKnobs.CoalitionMode == "top_k" {
		coalition = selectCoalitionTopK(stakes, rbKnobs.CoalitionTopK)
	} else {
		coalition = selectCoalition(stakes, totalStake, rbKnobs.CoalitionShare)
	}
	alphaRisk, betaRisk := coalitionMasses(coalition, stakes, baseWeights, totalStake, totalBase)

	var (
		lamSignalTargetF float64
		lamRiskTargetF   float64
		lamComboTargetF  float64
		riskBudgetSat    bool
		score            float64
	)

	if policyMode == "adaptive" {
		score = adaptiveScore(fNorm, gNorm, splitPressure, knobs)
		lamSignalTargetF = adaptiveSignalTarget(score, knobs)

		if rbKnobs.Enabled {
			lamRiskTargetF, riskBudgetSat = riskBudgetTarget(
				alphaRisk, betaRisk, uint64(msg.Size_),
				rbKnobs.Theta, rbKnobs.Epsilon, knobs.LamMax, rbKnobs.GridStep,
			)
		}

		switch rbKnobs.Mode {
		case "risk":
			lamComboTargetF = lamRiskTargetF
		case "hybrid":
			lamComboTargetF = math.Max(lamSignalTargetF, lamRiskTargetF)
		default: // "signal" or unrecognized
			lamComboTargetF = lamSignalTargetF
		}

		lamPrevF := float64(lamManual) / float64(S)
		riskOnly := rbKnobs.Mode == "risk"
		lamAutoF := adaptiveLambdaShell(lamPrevF, lamComboTargetF, score, knobs, riskOnly)

		lam = uint64(math.Round(lamAutoF * float64(S)))
		k.SetLambdaPpm(ctx, lam)
	} else {
		// Manual policy: freeze λ at the value previously set by MsgSetLambda.
		lam = lamManual
	}

	// Edge Case: if the active baseline degenerated to zero mass, fall back to
	// uniform so the mix is always well-defined.
	useUniformBaseline := (totalBase == 0)

	// Compute Integer Weights
	// Target: w_i = (1-λ)*p_i + λ*u_i, where u_i is the active baseline
	// (service_age | capped_stake | uniform), normalized by totalBase.

	weights := make([]uint64, N)
	var totalW uint64
	var attackerWeight uint64

	for i := range vals {
		stakePpm := ppmRatio(stakes[i], totalStake, S)

		// Base layer: active dual-path baseline (persistence or concentration).
		basePpm := uint64(0)
		if useUniformBaseline {
			if N > 0 {
				basePpm = S / uint64(N)
			}
		} else {
			basePpm = ppmRatio(baseWeights[i], totalBase, S)
		}

		// Adaptive layer: mix stake with the active baseline under lambda control.
		w, err := mixedWeightPpm(stakePpm, basePpm, lam, S)
		if err != nil {
			return nil, err
		}
		weights[i] = w
		var ok bool
		totalW, ok = checkedAdd(totalW, w)
		if !ok {
			return nil, fmt.Errorf("total weight overflow")
		}
		if isAttacker[i] {
			attackerWeight, ok = checkedAdd(attackerWeight, w)
			if !ok {
				return nil, fmt.Errorf("attacker weight overflow")
			}
		}
	}

	if totalW == 0 {
		return nil, fmt.Errorf("total sampling weight is zero")
	}

	// Committee draw. The sampling scheme is selected by COMMITTEE_DRAW_MODE
	// (see committeeDrawMode): "wr" is with-replacement i.i.d. multi-seat sortition,
	// under which the coalition's seat count is exactly Binomial(m, p_t(lambda)) and
	// the binomial certificate is the exact law of the draw (used by the risk-budget
	// assessment); "wor" is without-replacement sampling of distinct members, under
	// which the binomial is a conservative certificate (used by the adaptive-defence
	// experiments). Both modes share the mixed-weight construction above.
	// Tag is included so that multiple draws within the same block produce distinct committees.
	// Production note: tag is user-controlled — a production system should use a VRF or block hash instead.
	drawMode := committeeDrawMode()
	seedMaterial := fmt.Sprintf("chain=%s|h=%d|m=%d|n=%d|lam=%d|tag=%s", ctx.ChainID(), ctx.BlockHeight(), msg.Size_, N, lam, msg.Tag)
	seed := sha256.Sum256([]byte(seedMaterial))

	// Snapshot the pre-draw sampling weights for diagnostics: the without-replacement
	// loop zeroes weights[idx] for each selected validator, so emitting weights[i]
	// afterwards would log 0 for every member. Diagnostics need the mixed weight q_i.
	emittedWeights := make([]uint64, len(weights))
	copy(emittedWeights, weights)

	picked := make([]string, 0, msg.Size_)
	if drawMode == "wr" {
		// With replacement: each seat is an independent draw over the full set.
		for j := uint64(0); j < msg.Size_; j++ {
			seatHash := sha256.Sum256(append(seed[:], byte(j), byte(j>>8), byte(j>>16), byte(j>>24)))
			r := binary.BigEndian.Uint64(seatHash[:8]) % totalW
			var acc uint64
			idx := -1
			for i, w := range weights {
				nextAcc, ok := checkedAdd(acc, w)
				if !ok {
					return nil, fmt.Errorf("roulette accumulator overflow")
				}
				acc = nextAcc
				if r < acc {
					idx = i
					break
				}
			}
			if idx < 0 {
				return nil, fmt.Errorf("failed to sample committee member")
			}
			picked = append(picked, addrs[idx])
		}
	} else {
		// Without replacement: remove each selected validator from the pool.
		if int(msg.Size_) > N {
			return nil, fmt.Errorf("committee size %d exceeds validator set size %d (without-replacement)", msg.Size_, N)
		}
		remainingW := totalW
		for j := uint64(0); j < msg.Size_; j++ {
			if remainingW == 0 {
				return nil, fmt.Errorf("remaining sampling weight is zero before committee is filled")
			}
			seatHash := sha256.Sum256(append(seed[:], byte(j), byte(j>>8), byte(j>>16), byte(j>>24)))
			r := binary.BigEndian.Uint64(seatHash[:8]) % remainingW
			var acc uint64
			idx := -1
			for i, w := range weights {
				if w == 0 {
					continue
				}
				nextAcc, ok := checkedAdd(acc, w)
				if !ok {
					return nil, fmt.Errorf("roulette accumulator overflow")
				}
				acc = nextAcc
				if r < acc {
					idx = i
					break
				}
			}
			if idx < 0 {
				return nil, fmt.Errorf("failed to sample committee member")
			}
			picked = append(picked, addrs[idx])
			remainingW -= weights[idx]
			weights[idx] = 0 // remove selected validator (WOR)
		}
	}

	// Emit an event so the effect is visible in logs and can be parsed by demo scripts.
	// We include members as multiple attributes (member=...) for easy grep/JSON parsing.
	membersCsv := strings.Join(picked, ",")

	attackerStakePpm := ppmRatio(attackerStake, totalStake, S)
	attackerAgePpm := ppmRatio(attackerAge, totalAge, S)
	attackerWeightPpm := ppmRatio(attackerWeight, totalW, S)

	// Store a state-based payload for stable querying from runners (independent of tx index/events):
	// members_csv|a_stake=<ppm>|a_age=<ppm>|a_weight=<ppm>|a_vals=<n>|v_metrics=<addr:stake:age:weight;...>
	vparts := make([]string, 0, N)
	for i := range vals {
		vparts = append(vparts, fmt.Sprintf("%s:%d:%d:%d", addrs[i], stakes[i], ages[i], emittedWeights[i]))
	}
	vMetrics := strings.Join(vparts, ";")
	freshPpm := uint64(math.Round(freshPressure * float64(S)))
	giniPpm := uint64(math.Round(gini * float64(S)))

	// Risk-budget diagnostics: B_t(0) is the policy certificate at zero mixing
	// (the worst case under the current coalition); B_t(lambda_auto) is the
	// realized certificate after the controller acted.
	lamAutoFloat := float64(lam) / float64(S)
	boundAt0 := binomialCaptureTail(alphaRisk, betaRisk, uint64(msg.Size_), rbKnobs.Theta, 0)
	boundAtAuto := binomialCaptureTail(alphaRisk, betaRisk, uint64(msg.Size_), rbKnobs.Theta, lamAutoFloat)
	boundAt0Log10e6 := log10ToFixedPoint(boundAt0)
	boundAtAutoLog10e6 := log10ToFixedPoint(boundAtAuto)
	lamSignalTargetPpm := uint64(math.Round(lamSignalTargetF * float64(S)))
	lamRiskTargetPpm := uint64(math.Round(lamRiskTargetF * float64(S)))
	lamComboTargetPpm := uint64(math.Round(lamComboTargetF * float64(S)))
	riskAlphaPpm := uint64(math.Round(alphaRisk * float64(S)))
	riskBetaPpm := uint64(math.Round(betaRisk * float64(S)))
	riskSatInt := 0
	if riskBudgetSat {
		riskSatInt = 1
	}
	coalitionSize := len(coalition)

	lastDrawPayload := fmt.Sprintf(
		"%s|a_stake=%d|a_age=%d|a_weight=%d|a_vals=%d|v_metrics=%s|l_auto=%d|l_manual=%d|gini=%d|fresh=%d|policy=%s|l_signal=%d|l_risk=%d|l_target=%d|risk_alpha=%d|risk_beta=%d|risk_sat=%d|risk_ck=%d|risk_b0_log10e6=%d|risk_ba_log10e6=%d|rb_mode=%s",
		membersCsv, attackerStakePpm, attackerAgePpm, attackerWeightPpm, attackerCount, vMetrics, lam, lamManual, giniPpm, freshPpm, policyMode,
		lamSignalTargetPpm, lamRiskTargetPpm, lamComboTargetPpm, riskAlphaPpm, riskBetaPpm, riskSatInt, coalitionSize, boundAt0Log10e6, boundAtAutoLog10e6, rbKnobs.Mode,
	)
	k.SetLastDraw(ctx, msg.Tag, lastDrawPayload)

	attrs := []sdk.Attribute{
		sdk.NewAttribute("tag", msg.Tag),
		sdk.NewAttribute("size", fmt.Sprintf("%d", msg.Size_)),
		sdk.NewAttribute("lambda_ppm", fmt.Sprintf("%d", lam)),
		sdk.NewAttribute("lambda_manual_ppm", fmt.Sprintf("%d", lamManual)),
		sdk.NewAttribute("lambda_auto_ppm", fmt.Sprintf("%d", lam)),
		sdk.NewAttribute("policy_mode", policyMode),
		sdk.NewAttribute("gini_ppm", fmt.Sprintf("%d", giniPpm)),
		sdk.NewAttribute("fresh_pressure_ppm", fmt.Sprintf("%d", freshPpm)),
		sdk.NewAttribute("p_tau_max", fmt.Sprintf("%d", tauMax)),
		sdk.NewAttribute("p_zeta", fmt.Sprintf("%d", zetaPpm)),
		sdk.NewAttribute("a_vals", fmt.Sprintf("%d", attackerCount)),
		sdk.NewAttribute("a_stake", fmt.Sprintf("%d", attackerStakePpm)),
		sdk.NewAttribute("a_age", fmt.Sprintf("%d", attackerAgePpm)),
		sdk.NewAttribute("a_service", fmt.Sprintf("%d", attackerAgePpm)),
		sdk.NewAttribute("a_weight", fmt.Sprintf("%d", attackerWeightPpm)),
		sdk.NewAttribute("attacker_stake_ppm", fmt.Sprintf("%d", attackerStakePpm)),   // Legacy compat
		sdk.NewAttribute("attacker_age_ppm", fmt.Sprintf("%d", attackerAgePpm)),       // Kept key name; now service-age share
		sdk.NewAttribute("attacker_service_ppm", fmt.Sprintf("%d", attackerAgePpm)),   // Explicit semantic key
		sdk.NewAttribute("attacker_weight_ppm", fmt.Sprintf("%d", attackerWeightPpm)), // Legacy compat
		sdk.NewAttribute("members_csv", membersCsv),
		sdk.NewAttribute("lambda_signal_target_ppm", fmt.Sprintf("%d", lamSignalTargetPpm)),
		sdk.NewAttribute("lambda_risk_target_ppm", fmt.Sprintf("%d", lamRiskTargetPpm)),
		sdk.NewAttribute("lambda_target_ppm", fmt.Sprintf("%d", lamComboTargetPpm)),
		sdk.NewAttribute("risk_alpha_ppm", fmt.Sprintf("%d", riskAlphaPpm)),
		sdk.NewAttribute("risk_beta_ppm", fmt.Sprintf("%d", riskBetaPpm)),
		sdk.NewAttribute("risk_budget_satisfied", fmt.Sprintf("%d", riskSatInt)),
		sdk.NewAttribute("risk_coalition_size", fmt.Sprintf("%d", coalitionSize)),
		sdk.NewAttribute("risk_bound0_log10e6", fmt.Sprintf("%d", boundAt0Log10e6)),
		sdk.NewAttribute("risk_bound_auto_log10e6", fmt.Sprintf("%d", boundAtAutoLog10e6)),
		sdk.NewAttribute("risk_controller_mode", rbKnobs.Mode),
	}
	// ctx.Logger().Info("DrawCommittee emitting", "attrs", attrs)
	ctx.EventManager().EmitEvent(sdk.NewEvent("committee_drawn", attrs...))

	return &types.MsgDrawCommitteeResponse{}, nil
}

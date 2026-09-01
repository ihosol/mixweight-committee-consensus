package keeper

import (
	"math"
	"testing"
)

// TestSelectCoalition_MinShare covers the canonical small cases.
func TestSelectCoalition_MinShare(t *testing.T) {
	tests := []struct {
		name        string
		stakes      []uint64
		totalStake  uint64
		phi         float64
		wantSize    int
		wantHas     int // index that must be in the coalition (top stake)
	}{
		{
			name:       "one whale crosses threshold alone",
			stakes:     []uint64{100, 30, 20, 10, 5},
			totalStake: 165,
			phi:        0.33,
			wantSize:   1,
			wantHas:    0,
		},
		{
			name:       "top two needed to reach 1/3",
			stakes:     []uint64{30, 25, 20, 15, 10},
			totalStake: 100,
			phi:        0.5,
			wantSize:   2,
			wantHas:    0,
		},
		{
			name:       "uniform stake panel",
			stakes:     []uint64{20, 20, 20, 20, 20},
			totalStake: 100,
			phi:        0.4,
			wantSize:   2,
			wantHas:    0,
		},
		{
			name:       "phi 1.0 returns full set",
			stakes:     []uint64{50, 30, 20},
			totalStake: 100,
			phi:        1.0,
			wantSize:   3,
			wantHas:    0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := selectCoalition(tc.stakes, tc.totalStake, tc.phi)
			if len(got) != tc.wantSize {
				t.Fatalf("size: got %d want %d (got=%v)", len(got), tc.wantSize, got)
			}
			found := false
			for _, idx := range got {
				if idx == tc.wantHas {
					found = true
				}
			}
			if !found {
				t.Fatalf("expected index %d in coalition, got %v", tc.wantHas, got)
			}
		})
	}
}

func TestSelectCoalition_EmptyInputs(t *testing.T) {
	if got := selectCoalition(nil, 0, 0.33); got != nil {
		t.Fatalf("expected nil, got %v", got)
	}
	if got := selectCoalition([]uint64{10, 20}, 30, 0); got != nil {
		t.Fatalf("expected nil for phi=0, got %v", got)
	}
}

func TestCoalitionMasses_Basic(t *testing.T) {
	stakes := []uint64{60, 30, 10}
	ages := []uint64{600, 600, 200}
	totalStake := uint64(100)
	totalAge := uint64(1400)
	alpha, beta := coalitionMasses([]int{0}, stakes, ages, totalStake, totalAge)
	if math.Abs(alpha-0.6) > 1e-9 {
		t.Errorf("alpha: got %v want 0.6", alpha)
	}
	expectedBeta := 600.0 / 1400.0
	if math.Abs(beta-expectedBeta) > 1e-9 {
		t.Errorf("beta: got %v want %v", beta, expectedBeta)
	}
}

func TestBinomialUpperTail_Known(t *testing.T) {
	if got := binomialUpperTail(0.5, 9, 5); math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("P(X>=5|B(9,0.5)): got %.10f want 0.5", got)
	}
	if got := binomialUpperTail(0.3, 9, 0); got != 1 {
		t.Fatalf("q=0: got %v want 1", got)
	}
	if got := binomialUpperTail(0.9, 9, 10); got != 0 {
		t.Fatalf("q>m: got %v want 0", got)
	}
	if got := binomialUpperTail(0, 9, 1); got != 0 {
		t.Fatalf("p=0: got %v want 0", got)
	}
	if got := binomialUpperTail(1, 9, 9); math.Abs(got-1) > 1e-12 {
		t.Fatalf("p=1,q=m: got %v want 1", got)
	}
	prev := -1.0
	for p := 0.0; p <= 1.0+1e-9; p += 0.05 {
		cur := binomialUpperTail(p, 9, 5)
		if cur < prev-1e-12 {
			t.Fatalf("tail not increasing in p at p=%.2f: %.6f < %.6f", p, cur, prev)
		}
		prev = cur
	}
}

func TestBinomialCaptureTail_Known(t *testing.T) {
	if got := binomialCaptureTail(0.5, 0.2, 9, 0.5, 0); math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("B_t(0): got %.6f want 0.5", got)
	}
}

func TestBinomialCaptureTail_MonotonicInLambda(t *testing.T) {
	alpha, beta, m, theta := 0.5, 0.2, uint64(9), 0.5
	prev := binomialCaptureTail(alpha, beta, m, theta, 0)
	for lam := 0.05; lam <= 1.0+1e-9; lam += 0.05 {
		cur := binomialCaptureTail(alpha, beta, m, theta, lam)
		if cur > prev+1e-12 {
			t.Fatalf("B_t not non-increasing in lambda at %.2f: %.6f > %.6f", lam, cur, prev)
		}
		prev = cur
	}
}

func TestBinomialCaptureTail_ZeroMass(t *testing.T) {
	if got := binomialCaptureTail(0, 0, 9, 0.5, 0); got != 0 {
		t.Fatalf("zero mass: got %v want 0", got)
	}
}

func TestRiskBudgetTarget_AlreadySafe(t *testing.T) {
	lam, sat := riskBudgetTarget(0.3, 0.1, 9, 0.5, 0.15, 0.6, 0.01)
	if !sat || lam != 0 {
		t.Fatalf("already-safe: got lam=%v sat=%v want lam=0 sat=true", lam, sat)
	}
}

func TestRiskBudgetTarget_RiseRequired(t *testing.T) {
	alpha, beta, m, theta, eps, lmax, step := 0.5, 0.2, uint64(9), 0.5, 0.10, 0.8, 0.01
	lam, sat := riskBudgetTarget(alpha, beta, m, theta, eps, lmax, step)
	if !sat {
		t.Fatalf("expected feasible, got sat=false (lam=%v)", lam)
	}
	if b := binomialCaptureTail(alpha, beta, m, theta, lam); b > eps {
		t.Fatalf("B_t(%.3f)=%.4f > eps=%.2f", lam, b, eps)
	}
	if lam-step >= 0 {
		if b := binomialCaptureTail(alpha, beta, m, theta, lam-step); b <= eps {
			t.Fatalf("not minimal: B_t(%.3f)=%.4f already <= eps", lam-step, b)
		}
	}
}

func TestRiskBudgetTarget_BaselineDominated(t *testing.T) {
	alpha, beta, m, theta, eps := 0.3, 0.5, uint64(9), 0.5, 0.05
	lam, sat := riskBudgetTarget(alpha, beta, m, theta, eps, 0.6, 0.01)
	if sat || lam != 0 {
		t.Fatalf("baseline-dominated: got lam=%v sat=%v want lam=0 sat=false", lam, sat)
	}
}

func TestRiskBudgetTarget_NonMonotoneReturnsArgmin(t *testing.T) {
	alpha, beta, m, theta, eps, lmax, step := 0.6, 0.4, uint64(9), 0.5, 0.001, 0.3, 0.01
	lam, sat := riskBudgetTarget(alpha, beta, m, theta, eps, lmax, step)
	if sat {
		t.Fatalf("expected infeasible, got sat=true")
	}
	if math.Abs(lam-lmax) > step {
		t.Fatalf("argmin should be near lambdaMax=%.2f, got %.3f", lmax, lam)
	}
}

func TestRiskBudgetTarget_BadInputs(t *testing.T) {
	if _, sat := riskBudgetTarget(0.4, 0.1, 9, 0.5, 0, 0.5, 0.01); sat {
		t.Fatalf("eps<=0 must be infeasible")
	}
	if _, sat := riskBudgetTarget(0.4, 0.1, 9, 0.5, 1e-3, 0, 0.01); sat {
		t.Fatalf("lambdaMax<=0 must be infeasible")
	}
}

func TestBuildBaselineWeights_Uniform(t *testing.T) {
	stakes := []uint64{100, 50, 10}
	w, total := buildBaselineWeights("uniform", stakes, nil, 160, 0)
	if total != 3 {
		t.Fatalf("uniform total: got %d want 3", total)
	}
	for i, v := range w {
		if v != 1 {
			t.Errorf("uniform weight[%d]: got %d want 1", i, v)
		}
	}
}

func TestBuildBaselineWeights_ServiceAge(t *testing.T) {
	stakes := []uint64{100, 50}
	serviceAge := []uint64{600, 200}
	w, total := buildBaselineWeights("service_age", stakes, serviceAge, 150, 0)
	if total != 800 {
		t.Fatalf("service_age total: got %d want 800", total)
	}
	if w[0] != 600 || w[1] != 200 {
		t.Errorf("service_age weights: got %v want [600 200]", w)
	}
}

func TestBuildBaselineWeights_CappedStake_AntiConcentrationSign(t *testing.T) {
	// The decisive property: when a whale concentrates stake, capped_stake
	// must yield coalition baseline mass beta < coalition stake mass alpha.
	// Pre-drift: roughly uniform stakes -> beta ~= alpha.
	// Post-drift: top-2 hold most stake but cap their baseline -> beta < alpha.
	totalStake := uint64(1_000_000)
	capPpm := uint64(125000) // cap = 12.5% of total = 125000 tokens

	// Post-drift distribution: top-2 = 300k each, rest 8 share 400k = 50k each.
	stakes := []uint64{300000, 300000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000}
	w, total := buildBaselineWeights("capped_stake", stakes, nil, totalStake, capPpm)

	// top-2 must be capped at 125000.
	if w[0] != 125000 || w[1] != 125000 {
		t.Errorf("whales not capped: got w0=%d w1=%d want 125000", w[0], w[1])
	}
	// small validators uncapped.
	if w[2] != 50000 {
		t.Errorf("small validator capped wrongly: got %d want 50000", w[2])
	}

	coalition := []int{0, 1} // top-2
	var alphaStake uint64
	for _, idx := range coalition {
		alphaStake += stakes[idx]
	}
	alpha := float64(alphaStake) / float64(totalStake)
	var betaMass uint64
	for _, idx := range coalition {
		betaMass += w[idx]
	}
	beta := float64(betaMass) / float64(total)

	if !(beta < alpha) {
		t.Errorf("anti-concentration sign violated: beta=%.4f should be < alpha=%.4f", beta, alpha)
	}
	t.Logf("capped_stake post-drift: alpha=%.4f beta=%.4f (beta<alpha OK)", alpha, beta)
}

func TestBuildBaselineWeights_CappedStake_NoCapWhenUnderThreshold(t *testing.T) {
	// Pre-drift uniform stakes all below the cap -> capped_stake == raw stake
	// (degenerates to stake-proportional, i.e. beta == alpha; harmless).
	totalStake := uint64(1_200_000)
	capPpm := uint64(125000) // cap = 150000; all stakes are 100000 < cap
	stakes := make([]uint64, 12)
	for i := range stakes {
		stakes[i] = 100000
	}
	w, total := buildBaselineWeights("capped_stake", stakes, nil, totalStake, capPpm)
	for i, v := range w {
		if v != 100000 {
			t.Errorf("under-cap weight[%d]: got %d want 100000 (uncapped)", i, v)
		}
	}
	if total != 1_200_000 {
		t.Errorf("total: got %d want 1200000", total)
	}
}

func TestLog10ToFixedPoint(t *testing.T) {
	cases := []struct {
		in  float64
		out int64
	}{
		{1.0, 0},
		{0.1, -1_000_000},
		{0.01, -2_000_000},
		{1e-6, -6_000_000},
		{0, math.MinInt64},
		{-0.5, math.MinInt64},
	}
	for _, tc := range cases {
		if got := log10ToFixedPoint(tc.in); got != tc.out {
			t.Errorf("in=%v: got %v want %v", tc.in, got, tc.out)
		}
	}
}

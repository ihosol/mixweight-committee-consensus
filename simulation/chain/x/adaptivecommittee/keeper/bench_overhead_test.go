package keeper

// Benchmark of the per-epoch adaptive-controller overhead: the three signals
// (stake-Gini, freshness-dispersion, split-pressure), the asymmetric filter
// update, the service-age baseline construction, and the mixed-weight vector.
// This is exactly the extra work the x/adaptivecommittee module adds per epoch
// on top of an ordinary committee draw; the PPSWOR draw itself is excluded
// because any committee-based protocol pays it regardless of the controller.
//
// Run with:
//   go test -bench BenchmarkEpochUpdate -benchmem -run '^$' ./x/adaptivecommittee/keeper/

import (
	"math"
	"math/rand"
	"testing"
)

func benchEpochUpdate(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(42))
	stakes := make([]uint64, n)
	ages := make([]uint64, n)
	var total uint64
	for i := range stakes {
		stakes[i] = uint64(rng.Intn(1_000_000) + 1)
		ages[i] = uint64(rng.Intn(5_000))
		total += stakes[i]
	}
	knobs := adaptiveControllerKnobs{
		LamMax:              0.65,
		AlphaUp:             0.78,
		AlphaDown:           0.02,
		HysteresisFloor:     0.12,
		HysteresisTrigger:   0.18,
		FreshnessWeight:     0.75,
		GiniWeight:          0.10,
		SplitWeight:         0.15,
		FreshnessNormOffset: 0.03,
		FreshnessNormSpan:   0.22,
		GiniNormOffset:      0.18,
		GiniNormSpan:        0.20,
	}
	const S = uint64(1_000_000)
	const horizon = 40.0
	const zeta = 0.5
	const tauMax = 2_000.0
	baseW := make([]uint64, n)
	lamPrev := 0.25

	b.ReportAllocs()
	b.ResetTimer()
	for it := 0; it < b.N; it++ {
		// Signals.
		g := giniFromStakes(stakes)
		f := freshPressureFromAge(stakes, ages, total, horizon)
		var freshCount int
		var freshStake uint64
		for i := range ages {
			if float64(ages[i]) < horizon {
				freshCount++
				freshStake += stakes[i]
			}
		}
		split := clamp01(float64(freshCount)/float64(n) - float64(freshStake)/float64(total))

		// Controller update.
		fNorm := clamp01((f - knobs.FreshnessNormOffset) / knobs.FreshnessNormSpan)
		gNorm := clamp01((g - knobs.GiniNormOffset) / knobs.GiniNormSpan)
		score := adaptiveScore(fNorm, gNorm, split, knobs)
		lam := adaptiveLambdaNext(lamPrev, score, knobs)
		lamPpm := uint64(math.Round(lam * float64(S)))

		// Service-age baseline: concave capped transform times stake, then normalize.
		var totalBase uint64
		for i := range stakes {
			tau := math.Min(float64(ages[i]), tauMax)
			tt := math.Max(1, math.Pow(tau, zeta))
			bw := uint64(tt * float64(stakes[i]) / 1_000.0)
			baseW[i] = bw
			totalBase += bw
		}

		// Mixed-weight vector.
		var acc uint64
		for i := range stakes {
			stakePpm := ppmRatio(stakes[i], total, S)
			basePpm := ppmRatio(baseW[i], totalBase, S)
			w, err := mixedWeightPpm(stakePpm, basePpm, lamPpm, S)
			if err != nil {
				b.Fatal(err)
			}
			acc += w
		}
		if acc == 0 {
			b.Fatal("degenerate weight vector")
		}
		lamPrev = lam
	}
}

func BenchmarkEpochUpdateN100(b *testing.B)    { benchEpochUpdate(b, 100) }
func BenchmarkEpochUpdateN1000(b *testing.B)   { benchEpochUpdate(b, 1_000) }
func BenchmarkEpochUpdateN10000(b *testing.B)  { benchEpochUpdate(b, 10_000) }
func BenchmarkEpochUpdateN100000(b *testing.B) { benchEpochUpdate(b, 100_000) }

package keeper

import (
	"encoding/binary"
	"math"

	"chain-five-three/x/adaptivecommittee/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
)

const (
	lambdaScalePpm uint64 = 1_000_000

	// PoC defaults for persistence transform: \tilde{tau}=min(tau,tau_max)^zeta.
	defaultPersistenceTauMaxBlocks uint64 = 2_000
	defaultPersistenceZetaPpm      uint64 = 500_000 // sqrt-like concavity (zeta=0.5)

	// Adaptive controller defaults (ppm-scaled floats in [0, 1_000_000]).
	defaultAdaptiveLamMaxPpm            uint64 = 550_000
	defaultAdaptiveAlphaUpPpm           uint64 = 780_000
	defaultAdaptiveAlphaDownPpm         uint64 = 60_000
	defaultAdaptiveHysteresisFloorPpm   uint64 = 120_000
	defaultAdaptiveHysteresisTriggerPpm uint64 = 180_000
	defaultAdaptiveFreshnessWeightPpm   uint64 = 750_000
	defaultAdaptiveGiniWeightPpm        uint64 = 100_000
	defaultAdaptiveSplitWeightPpm       uint64 = 150_000

	// Affine normalizer defaults for the freshness-dispersion and stake-Gini
	// signals. These correspond to F0=0.03, Fspan=0.22, G0=0.18, Gspan=0.20 in
	// the ppm-scaled [0, 1_000_000] range and match the values tabulated in the
	// dissertation's Chapter 3 (service-age controller). Span values must be
	// strictly positive; a zero-span configuration falls back to its default.
	defaultAdaptiveFreshnessNormOffsetPpm uint64 = 30_000
	defaultAdaptiveFreshnessNormSpanPpm   uint64 = 220_000
	defaultAdaptiveGiniNormOffsetPpm      uint64 = 180_000
	defaultAdaptiveGiniNormSpanPpm        uint64 = 200_000

	// Freshness horizon (in blocks): defines what "fresh" means temporally
	// for both the freshness-dispersion signal and the Sybil-blind split-pressure signal.
	defaultAdaptiveFreshnessHorizonBlocks uint64 = 40
)

func (k Keeper) GetLambdaPpm(ctx sdk.Context) uint64 {
	store := ctx.KVStore(k.storeKey)
	bz := store.Get(types.KeyPrefix(types.LambdaPpmKey))
	if len(bz) == 0 {
		return 0
	}
	return binary.BigEndian.Uint64(bz)
}

func (k Keeper) SetLambdaPpm(ctx sdk.Context, v uint64) {
	if v > lambdaScalePpm {
		v = lambdaScalePpm
	}
	store := ctx.KVStore(k.storeKey)
	bz := make([]byte, 8)
	binary.BigEndian.PutUint64(bz, v)
	store.Set(types.KeyPrefix(types.LambdaPpmKey), bz)
}

func (k Keeper) SetLastDraw(ctx sdk.Context, tag string, membersCsv string) {
	store := ctx.KVStore(k.storeKey)
	store.Set(append(types.KeyPrefix(types.LastDrawPrefix), []byte(tag)...), []byte(membersCsv))
}

func (k Keeper) GetLastDraw(ctx sdk.Context, tag string) (string, bool) {
	store := ctx.KVStore(k.storeKey)
	bz := store.Get(append(types.KeyPrefix(types.LastDrawPrefix), []byte(tag)...))
	if bz == nil {
		return "", false
	}
	return string(bz), true
}

func (k Keeper) GetPersistenceTauMaxBlocks(ctx sdk.Context) uint64 {
	store := ctx.KVStore(k.storeKey)
	bz := store.Get(types.KeyPrefix(types.PersistenceTauMaxBlocksKey))
	if len(bz) == 0 {
		return defaultPersistenceTauMaxBlocks
	}
	return binary.BigEndian.Uint64(bz)
}

func (k Keeper) GetPersistenceZetaPpm(ctx sdk.Context) uint64 {
	store := ctx.KVStore(k.storeKey)
	bz := store.Get(types.KeyPrefix(types.PersistenceZetaPpmKey))
	if len(bz) == 0 {
		return defaultPersistenceZetaPpm
	}
	v := binary.BigEndian.Uint64(bz)
	if v > lambdaScalePpm {
		return lambdaScalePpm
	}
	return v
}

func (k Keeper) getUint64OrDefault(ctx sdk.Context, key string, def uint64) uint64 {
	store := ctx.KVStore(k.storeKey)
	bz := store.Get(types.KeyPrefix(key))
	if len(bz) == 0 {
		return def
	}
	v := binary.BigEndian.Uint64(bz)
	if v > lambdaScalePpm {
		return lambdaScalePpm
	}
	return v
}

func (k Keeper) GetAdaptiveLamMaxPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveLamMaxPpmKey, defaultAdaptiveLamMaxPpm)
}
func (k Keeper) GetAdaptiveAlphaUpPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveAlphaUpPpmKey, defaultAdaptiveAlphaUpPpm)
}
func (k Keeper) GetAdaptiveAlphaDownPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveAlphaDownPpmKey, defaultAdaptiveAlphaDownPpm)
}
func (k Keeper) GetAdaptiveHysteresisFloorPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveHysteresisFloorPpmKey, defaultAdaptiveHysteresisFloorPpm)
}
func (k Keeper) GetAdaptiveHysteresisTriggerPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveHysteresisTriggerPpmKey, defaultAdaptiveHysteresisTriggerPpm)
}
func (k Keeper) GetAdaptiveFreshnessWeightPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveFreshnessWeightPpmKey, defaultAdaptiveFreshnessWeightPpm)
}
func (k Keeper) GetAdaptiveGiniWeightPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveGiniWeightPpmKey, defaultAdaptiveGiniWeightPpm)
}
func (k Keeper) GetAdaptiveSplitWeightPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveSplitWeightPpmKey, defaultAdaptiveSplitWeightPpm)
}

func (k Keeper) GetAdaptiveFreshnessNormOffsetPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveFreshnessNormOffsetPpmKey, defaultAdaptiveFreshnessNormOffsetPpm)
}

func (k Keeper) GetAdaptiveFreshnessNormSpanPpm(ctx sdk.Context) uint64 {
	v := k.getUint64OrDefault(ctx, types.AdaptiveFreshnessNormSpanPpmKey, defaultAdaptiveFreshnessNormSpanPpm)
	if v == 0 {
		return defaultAdaptiveFreshnessNormSpanPpm
	}
	return v
}

func (k Keeper) GetAdaptiveGiniNormOffsetPpm(ctx sdk.Context) uint64 {
	return k.getUint64OrDefault(ctx, types.AdaptiveGiniNormOffsetPpmKey, defaultAdaptiveGiniNormOffsetPpm)
}

func (k Keeper) GetAdaptiveGiniNormSpanPpm(ctx sdk.Context) uint64 {
	v := k.getUint64OrDefault(ctx, types.AdaptiveGiniNormSpanPpmKey, defaultAdaptiveGiniNormSpanPpm)
	if v == 0 {
		return defaultAdaptiveGiniNormSpanPpm
	}
	return v
}

func (k Keeper) GetAdaptiveFreshnessHorizonBlocks(ctx sdk.Context) uint64 {
	store := ctx.KVStore(k.storeKey)
	bz := store.Get(types.KeyPrefix(types.AdaptiveFreshnessHorizonBlocksKey))
	if len(bz) == 0 {
		return defaultAdaptiveFreshnessHorizonBlocks
	}
	v := binary.BigEndian.Uint64(bz)
	if v == 0 {
		return defaultAdaptiveFreshnessHorizonBlocks
	}
	return v
}

func (k Keeper) ApplyPersistenceTransform(age uint64, tauMax uint64, zetaPpm uint64) uint64 {
	if tauMax > 0 && age > tauMax {
		age = tauMax
	}
	if age == 0 {
		return 0
	}
	if zetaPpm >= lambdaScalePpm {
		return age
	}
	if zetaPpm == 0 {
		return 1
	}
	// Fast exact path: zeta=0.5 -> integer sqrt.
	if zetaPpm == 500_000 {
		return intSqrt(age)
	}
	// General path: age^zeta via deterministic IEEE-754 math.Pow.
	// Go's math.Pow is platform-deterministic (pure-Go software implementation).
	zeta := float64(zetaPpm) / float64(lambdaScalePpm)
	v := math.Pow(float64(age), zeta)
	if v < 1.0 {
		return 1
	}
	// Round-half-away-from-zero for deterministic rounding across platforms.
	r := uint64(math.Floor(v + 0.5))
	if r == 0 {
		return 1
	}
	return r
}

func intSqrt(x uint64) uint64 {
	if x == 0 {
		return 0
	}
	z := x
	y := (z + 1) / 2
	for y < z {
		z = y
		y = (z + x/z) / 2
	}
	if z == 0 {
		return 1
	}
	return z
}

func LambdaScalePpm() uint64 { return lambdaScalePpm }

package types

const (
	// ModuleName defines the module name
	ModuleName = "adaptivecommittee"

	// State keys (PoC). Values are stored in the module KVStore.
	// Lambda is stored as an integer in parts-per-million: lambda_ppm in [0, 1_000_000].
	LambdaPpmKey = "lambda_ppm"

	// LastDrawPrefix stores the last drawn committee for a given tag.
	// Key: last_draw/<tag>  Value: members_csv (bytes)
	LastDrawPrefix = "last_draw/"

	// ValidatorFirstSeenPrefix stores the first seen block-height per validator operator.
	// Key: validator_first_seen/<operator_address>  Value: uint64(block_height) big-endian
	ValidatorFirstSeenPrefix = "validator_first_seen/"

	// Persistence parameters for the age baseline transform.
	// tau_max_blocks: cap before concave transform; zeta_ppm in [0, 1_000_000].
	PersistenceTauMaxBlocksKey = "persistence_tau_max_blocks"
	PersistenceZetaPpmKey      = "persistence_zeta_ppm"

	// Adaptive controller knobs (PoC, keeper-backed until formalized in params/proto).
	AdaptiveLamMaxPpmKey            = "adaptive_lam_max_ppm"
	AdaptiveAlphaUpPpmKey           = "adaptive_alpha_up_ppm"
	AdaptiveAlphaDownPpmKey         = "adaptive_alpha_down_ppm"
	AdaptiveHysteresisFloorPpmKey   = "adaptive_hysteresis_floor_ppm"
	AdaptiveHysteresisTriggerPpmKey = "adaptive_hysteresis_trigger_ppm"
	AdaptiveFreshnessWeightPpmKey   = "adaptive_freshness_weight_ppm"
	AdaptiveGiniWeightPpmKey        = "adaptive_gini_weight_ppm"
	AdaptiveSplitWeightPpmKey       = "adaptive_split_weight_ppm"

	// Affine normalizers for the freshness-dispersion and stake-Gini signals:
	// f_t = clip( (F_t - F0) / Fspan , 0, 1 ), g_t = clip( (G_t - G0) / Gspan , 0, 1 ).
	// Encoded in ppm so that offset/span both live on the same [0, 1_000_000] scale as
	// the other adaptive knobs. Span keys must be strictly positive at use-site.
	AdaptiveFreshnessNormOffsetPpmKey = "adaptive_freshness_norm_offset_ppm"
	AdaptiveFreshnessNormSpanPpmKey   = "adaptive_freshness_norm_span_ppm"
	AdaptiveGiniNormOffsetPpmKey      = "adaptive_gini_norm_offset_ppm"
	AdaptiveGiniNormSpanPpmKey        = "adaptive_gini_norm_span_ppm"

	// Freshness horizon (in blocks): temporal threshold for freshness-dispersion
	// and Sybil-blind split-pressure signals.
	AdaptiveFreshnessHorizonBlocksKey = "adaptive_freshness_horizon_blocks"

	// StoreKey defines the primary module store key
	StoreKey = ModuleName

	// RouterKey defines the module's message routing key
	RouterKey = ModuleName

	// MemStoreKey defines the in-memory store key
	MemStoreKey = "mem_adaptivecommittee"
)

func KeyPrefix(p string) []byte {
	return []byte(p)
}

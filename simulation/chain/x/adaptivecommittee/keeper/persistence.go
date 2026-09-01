package keeper

import (
	"encoding/binary"

	"chain-five-three/x/adaptivecommittee/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
)

func (k Keeper) GetValidatorFirstSeenHeight(ctx sdk.Context, operator string) (uint64, bool) {
	store := ctx.KVStore(k.storeKey)
	key := append(types.KeyPrefix(types.ValidatorFirstSeenPrefix), []byte(operator)...)
	bz := store.Get(key)
	if len(bz) == 0 {
		return 0, false
	}
	return binary.BigEndian.Uint64(bz), true
}

func (k Keeper) SetValidatorFirstSeenHeight(ctx sdk.Context, operator string, h uint64) {
	store := ctx.KVStore(k.storeKey)
	key := append(types.KeyPrefix(types.ValidatorFirstSeenPrefix), []byte(operator)...)
	bz := make([]byte, 8)
	binary.BigEndian.PutUint64(bz, h)
	store.Set(key, bz)
}

func (k Keeper) ObserveBondedValidators(ctx sdk.Context) {
	height := uint64(1)
	if ctx.BlockHeight() > 0 {
		height = uint64(ctx.BlockHeight())
	}

	vals, err := k.stakingKeeper.GetBondedValidatorsByPower(ctx)
	if err != nil {
		k.Logger(ctx).Error("ObserveBondedValidators failed", "err", err)
		return
	}
	for _, v := range vals {
		op := v.GetOperator()
		if _, ok := k.GetValidatorFirstSeenHeight(ctx, op); !ok {
			k.SetValidatorFirstSeenHeight(ctx, op, height)
		}
	}
}

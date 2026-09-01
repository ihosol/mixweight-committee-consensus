package keeper

import (
	"testing"

	"chain-five-three/x/adaptivecommittee/keeper"
	adaptivetypes "chain-five-three/x/adaptivecommittee/types"
	"cosmossdk.io/log"
	"cosmossdk.io/store/metrics"
	"cosmossdk.io/store/rootmulti"
	storetypes "cosmossdk.io/store/types"
	cmtproto "github.com/cometbft/cometbft/proto/tendermint/types"
	dbm "github.com/cosmos/cosmos-db"
	"github.com/cosmos/cosmos-sdk/codec"
	codectypes "github.com/cosmos/cosmos-sdk/codec/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
	paramtypes "github.com/cosmos/cosmos-sdk/x/params/types"
	"github.com/stretchr/testify/require"
)

func AdaptivecommitteeKeeper(t testing.TB) (*keeper.Keeper, sdk.Context) {
	t.Helper()

	storeKey := storetypes.NewKVStoreKey(adaptivetypes.StoreKey)
	memStoreKey := storetypes.NewMemoryStoreKey(adaptivetypes.MemStoreKey)
	paramsKey := storetypes.NewKVStoreKey(paramtypes.StoreKey)
	paramsTKey := storetypes.NewTransientStoreKey(paramtypes.TStoreKey)

	db := dbm.NewMemDB()
	stateStore := rootmulti.NewStore(db, log.NewNopLogger(), metrics.NewNoOpMetrics())
	stateStore.MountStoreWithDB(storeKey, storetypes.StoreTypeIAVL, db)
	stateStore.MountStoreWithDB(memStoreKey, storetypes.StoreTypeMemory, nil)
	stateStore.MountStoreWithDB(paramsKey, storetypes.StoreTypeIAVL, db)
	stateStore.MountStoreWithDB(paramsTKey, storetypes.StoreTypeTransient, nil)
	require.NoError(t, stateStore.LoadLatestVersion())

	ctx := sdk.NewContext(stateStore, cmtproto.Header{}, false, log.NewNopLogger())
	interfaceRegistry := codectypes.NewInterfaceRegistry()
	cdc := codec.NewProtoCodec(interfaceRegistry)
	legacyAmino := codec.NewLegacyAmino()
	paramsSubspace := paramtypes.NewSubspace(cdc, legacyAmino, paramsKey, paramsTKey, adaptivetypes.ModuleName)
	paramsSubspace = paramsSubspace.WithKeyTable(adaptivetypes.ParamKeyTable())

	k := keeper.NewKeeper(
		cdc,
		storeKey,
		memStoreKey,
		paramsSubspace,
		nil,
		nil,
		nil,
	)

	return k, ctx
}

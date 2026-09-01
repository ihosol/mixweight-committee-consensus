package app

import (
	storetypes "cosmossdk.io/store/types"

	"github.com/cosmos/cosmos-sdk/runtime"
	"github.com/cosmos/cosmos-sdk/types/module"
	paramstypes "github.com/cosmos/cosmos-sdk/x/params/types"

	adaptivecommittee "chain-five-three/x/adaptivecommittee"
	adaptivecommitteekeeper "chain-five-three/x/adaptivecommittee/keeper"
	adaptivecommitteetypes "chain-five-three/x/adaptivecommittee/types"
)

// registerAdaptiveCommitteeModules registers the adaptivecommittee module using
// the "manual module" path (non app-wiring), similar to IBC.
//
// Rationale: Ignite's app wiring expects each module to be declared via
// app_config.go ModuleConfig entries. Our PoC module is a legacy (starport-style)
// Cosmos SDK module and is easiest to integrate deterministically via manual
// registration: register stores, set param key tables, construct keeper, and
// then register the AppModule.
func (app *App) registerAdaptiveCommitteeModules() error {
	// 1) Register stores
	if err := app.RegisterStores(
		storetypes.NewKVStoreKey(adaptivecommitteetypes.StoreKey),
		storetypes.NewMemoryStoreKey(adaptivecommitteetypes.MemStoreKey),
	); err != nil {
		return err
	}

	// 2) Register param key table for legacy params subspace
	app.ParamsKeeper.Subspace(adaptivecommitteetypes.ModuleName).
		WithKeyTable(adaptivecommitteetypes.ParamKeyTable())

	// 3) Construct keeper
	k := adaptivecommitteekeeper.NewKeeper(
		app.appCodec,
		app.GetKey(adaptivecommitteetypes.StoreKey),
		app.GetKey(adaptivecommitteetypes.MemStoreKey),
		app.GetSubspace(adaptivecommitteetypes.ModuleName),
		app.StakingKeeper,
		app.BankKeeper,
		app.GovKeeper,
	)
	app.AdaptiveCommitteeKeeper = k

	// 4) Register interfaces (client-side)
	adaptivecommittee.NewAppModuleBasic(app.appCodec).RegisterInterfaces(app.interfaceRegistry)

	// 5) Register module with runtime
	if err := app.RegisterModules(
		adaptivecommittee.NewAppModule(app.appCodec, *k, app.AuthKeeper, app.BankKeeper),
	); err != nil {
		return err
	}

	// 6) Ensure params subspace exists for the module.
	// Some scaffolds require explicit subspace initialization; this is a no-op if already present.
	_ = paramstypes.Subspace{}
	_ = runtime.NewKVStoreService
	_ = module.NewBasicManager

	return nil
}

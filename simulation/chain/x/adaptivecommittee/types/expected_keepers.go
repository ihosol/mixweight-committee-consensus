package types

import (
	"context"

	sdk "github.com/cosmos/cosmos-sdk/types"
	stakingtypes "github.com/cosmos/cosmos-sdk/x/staking/types"
)

type StakingKeeper interface {
	// Minimal interface required for the PoC.
	// We use the bonded validator set ordered by power as the source distribution.
	GetBondedValidatorsByPower(ctx context.Context) ([]stakingtypes.Validator, error)
}

type GovKeeper interface {
	// Methods imported from gov should be defined here
}

// AccountKeeper defines the expected account keeper used by the module.
type AccountKeeper interface {
	GetAccount(ctx context.Context, addr sdk.AccAddress) sdk.AccountI
}

// BankKeeper defines the expected interface needed to retrieve account balances.
type BankKeeper interface {
	SpendableCoins(ctx context.Context, addr sdk.AccAddress) sdk.Coins
}

//go:build simulation

package simulation

import (
	"math/rand"

	"chain-five-three/x/adaptivecommittee/keeper"
	"chain-five-three/x/adaptivecommittee/types"
	"github.com/cosmos/cosmos-sdk/baseapp"
	sdk "github.com/cosmos/cosmos-sdk/types"
	simtypes "github.com/cosmos/cosmos-sdk/types/simulation"
)

func SimulateMsgSetLambda(
	ak types.AccountKeeper,
	bk types.BankKeeper,
	k keeper.Keeper,
) simtypes.Operation {
	return func(r *rand.Rand, app *baseapp.BaseApp, ctx sdk.Context, accs []simtypes.Account, chainID string,
	) (simtypes.OperationMsg, []simtypes.FutureOperation, error) {
		simAccount, _ := simtypes.RandomAcc(r, accs)
		msg := &types.MsgSetLambda{
			Creator: simAccount.Address.String(),
		}

		// TODO: Handling the SetLambda simulation

		return simtypes.NoOpMsg(types.ModuleName, msg.Type(), "SetLambda simulation not implemented"), nil, nil
	}
}

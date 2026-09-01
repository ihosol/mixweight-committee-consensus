//go:build simulation

package adaptivecommittee

import (
	"math/rand"

	"chain-five-three/testutil/sample"
	adaptivecommitteesimulation "chain-five-three/x/adaptivecommittee/simulation"
	"chain-five-three/x/adaptivecommittee/types"
	"github.com/cosmos/cosmos-sdk/baseapp"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/cosmos/cosmos-sdk/types/module"
	simtypes "github.com/cosmos/cosmos-sdk/types/simulation"
	"github.com/cosmos/cosmos-sdk/x/simulation"
)

// avoid unused import issue
var (
	_ = sample.AccAddress
	_ = adaptivecommitteesimulation.FindAccount
	_ = simulation.MsgEntryKind
	_ = baseapp.Paramspace
	_ = rand.Rand{}
)

const (
	opWeightMsgDrawCommittee = "op_weight_msg_draw_committee"
	// TODO: Determine the simulation weight value
	defaultWeightMsgDrawCommittee int = 100

	opWeightMsgSetLambda = "op_weight_msg_set_lambda"
	// TODO: Determine the simulation weight value
	defaultWeightMsgSetLambda int = 100

	// this line is used by starport scaffolding # simapp/module/const
)

// GenerateGenesisState creates a randomized GenState of the module.
func (AppModule) GenerateGenesisState(simState *module.SimulationState) {
	accs := make([]string, len(simState.Accounts))
	for i, acc := range simState.Accounts {
		accs[i] = acc.Address.String()
	}
	adaptivecommitteeGenesis := types.GenesisState{
		Params: types.DefaultParams(),
		// this line is used by starport scaffolding # simapp/module/genesisState
	}
	simState.GenState[types.ModuleName] = simState.Cdc.MustMarshalJSON(&adaptivecommitteeGenesis)
}

// RegisterStoreDecoder registers a decoder.
func (am AppModule) RegisterStoreDecoder(_ sdk.StoreDecoderRegistry) {}

// ProposalContents doesn't return any content functions for governance proposals.
func (AppModule) ProposalContents(_ module.SimulationState) []simtypes.WeightedProposalContent {
	return nil
}

// WeightedOperations returns the all the gov module operations with their respective weights.
func (am AppModule) WeightedOperations(simState module.SimulationState) []simtypes.WeightedOperation {
	operations := make([]simtypes.WeightedOperation, 0)

	var weightMsgDrawCommittee int
	simState.AppParams.GetOrGenerate(simState.Cdc, opWeightMsgDrawCommittee, &weightMsgDrawCommittee, nil,
		func(_ *rand.Rand) {
			weightMsgDrawCommittee = defaultWeightMsgDrawCommittee
		},
	)
	operations = append(operations, simulation.NewWeightedOperation(
		weightMsgDrawCommittee,
		adaptivecommitteesimulation.SimulateMsgDrawCommittee(am.accountKeeper, am.bankKeeper, am.keeper),
	))

	var weightMsgSetLambda int
	simState.AppParams.GetOrGenerate(simState.Cdc, opWeightMsgSetLambda, &weightMsgSetLambda, nil,
		func(_ *rand.Rand) {
			weightMsgSetLambda = defaultWeightMsgSetLambda
		},
	)
	operations = append(operations, simulation.NewWeightedOperation(
		weightMsgSetLambda,
		adaptivecommitteesimulation.SimulateMsgSetLambda(am.accountKeeper, am.bankKeeper, am.keeper),
	))

	// this line is used by starport scaffolding # simapp/module/operation

	return operations
}

// ProposalMsgs returns msgs used for governance proposals for simulations.
func (am AppModule) ProposalMsgs(simState module.SimulationState) []simtypes.WeightedProposalMsg {
	return []simtypes.WeightedProposalMsg{
		simulation.NewWeightedProposalMsg(
			opWeightMsgDrawCommittee,
			defaultWeightMsgDrawCommittee,
			func(r *rand.Rand, ctx sdk.Context, accs []simtypes.Account) sdk.Msg {
				adaptivecommitteesimulation.SimulateMsgDrawCommittee(am.accountKeeper, am.bankKeeper, am.keeper)
				return nil
			},
		),
		simulation.NewWeightedProposalMsg(
			opWeightMsgSetLambda,
			defaultWeightMsgSetLambda,
			func(r *rand.Rand, ctx sdk.Context, accs []simtypes.Account) sdk.Msg {
				adaptivecommitteesimulation.SimulateMsgSetLambda(am.accountKeeper, am.bankKeeper, am.keeper)
				return nil
			},
		),
		// this line is used by starport scaffolding # simapp/module/OpMsg
	}
}

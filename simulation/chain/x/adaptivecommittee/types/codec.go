package types

import (
	"github.com/cosmos/cosmos-sdk/codec"
	cdctypes "github.com/cosmos/cosmos-sdk/codec/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
)

func RegisterCodec(cdc *codec.LegacyAmino) {
	cdc.RegisterConcrete(&MsgDrawCommittee{}, "adaptivecommittee/DrawCommittee", nil)
	cdc.RegisterConcrete(&MsgSetLambda{}, "adaptivecommittee/SetLambda", nil)
	// this line is used by starport scaffolding # 2
}

func RegisterInterfaces(registry cdctypes.InterfaceRegistry) {
	registry.RegisterImplementations((*sdk.Msg)(nil),
		&MsgDrawCommittee{},
	)
	registry.RegisterImplementations((*sdk.Msg)(nil),
		&MsgSetLambda{},
	)
	// this line is used by starport scaffolding # 3

	// NOTE: SDK v0.53 runtime validates cosmos.msg.v1 options on registered Msg service
	// descriptors. The legacy Starport-generated gogo descriptors for this module do not
	// carry those options in a runtime-visible form, so registering the service descriptor
	// here causes startup panics. We rely on explicit sdk.Msg implementations (GetSigners)
	// and module service registration instead.
	// msgservice.RegisterMsgServiceDesc(registry, &_Msg_serviceDesc)
}

var (
	Amino     = codec.NewLegacyAmino()
	ModuleCdc = codec.NewProtoCodec(cdctypes.NewInterfaceRegistry())
)

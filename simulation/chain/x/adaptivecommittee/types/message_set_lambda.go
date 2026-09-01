package types

import (
	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkerrors "github.com/cosmos/cosmos-sdk/types/errors"
)

const TypeMsgSetLambda = "set_lambda"

var _ sdk.Msg = &MsgSetLambda{}

func NewMsgSetLambda(creator string, lambdaPpm uint64) *MsgSetLambda {
	return &MsgSetLambda{
		Creator:   creator,
		LambdaPpm: lambdaPpm,
	}
}

func (msg *MsgSetLambda) Route() string {
	return RouterKey
}

func (msg *MsgSetLambda) Type() string {
	return TypeMsgSetLambda
}

func (msg *MsgSetLambda) GetSigners() []sdk.AccAddress {
	creator, err := sdk.AccAddressFromBech32(msg.Creator)
	if err != nil {
		panic(err)
	}
	return []sdk.AccAddress{creator}
}

func (msg *MsgSetLambda) GetSignBytes() []byte {
	bz := ModuleCdc.MustMarshalJSON(msg)
	return sdk.MustSortJSON(bz)
}

func (msg *MsgSetLambda) ValidateBasic() error {
	_, err := sdk.AccAddressFromBech32(msg.Creator)
	if err != nil {
		return errorsmod.Wrapf(sdkerrors.ErrInvalidAddress, "invalid creator address (%s)", err)
	}
	return nil
}

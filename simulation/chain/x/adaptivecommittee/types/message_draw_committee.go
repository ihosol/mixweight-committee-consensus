package types

import (
	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkerrors "github.com/cosmos/cosmos-sdk/types/errors"
)

const TypeMsgDrawCommittee = "draw_committee"

var _ sdk.Msg = &MsgDrawCommittee{}

func NewMsgDrawCommittee(creator string, size uint64, tag string) *MsgDrawCommittee {
	return &MsgDrawCommittee{
		Creator: creator,
		Size_:   size,
		Tag:     tag,
	}
}

func (msg *MsgDrawCommittee) Route() string {
	return RouterKey
}

func (msg *MsgDrawCommittee) Type() string {
	return TypeMsgDrawCommittee
}

func (msg *MsgDrawCommittee) GetSigners() []sdk.AccAddress {
	creator, err := sdk.AccAddressFromBech32(msg.Creator)
	if err != nil {
		panic(err)
	}
	return []sdk.AccAddress{creator}
}

func (msg *MsgDrawCommittee) GetSignBytes() []byte {
	bz := ModuleCdc.MustMarshalJSON(msg)
	return sdk.MustSortJSON(bz)
}

func (msg *MsgDrawCommittee) ValidateBasic() error {
	_, err := sdk.AccAddressFromBech32(msg.Creator)
	if err != nil {
		return errorsmod.Wrapf(sdkerrors.ErrInvalidAddress, "invalid creator address (%s)", err)
	}
	return nil
}

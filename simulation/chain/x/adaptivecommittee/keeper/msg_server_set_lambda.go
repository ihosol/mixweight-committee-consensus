package keeper

import (
	"context"
	"fmt"

	"chain-five-three/x/adaptivecommittee/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
)

func (k msgServer) SetLambda(goCtx context.Context, msg *types.MsgSetLambda) (*types.MsgSetLambdaResponse, error) {
	ctx := sdk.UnwrapSDKContext(goCtx)

	// PoC: allow any sender to set λ. In a realistic integration, restrict this to gov.
	S := LambdaScalePpm()
	if msg.LambdaPpm > S {
		return nil, fmt.Errorf("lambda_ppm must be <= %d", S)
	}
	k.SetLambdaPpm(ctx, msg.LambdaPpm)

	ctx.EventManager().EmitEvent(sdk.NewEvent(
		"lambda_updated",
		sdk.NewAttribute("lambda_ppm", fmt.Sprintf("%d", msg.LambdaPpm)),
		sdk.NewAttribute("creator", msg.Creator),
	))

	return &types.MsgSetLambdaResponse{}, nil
}

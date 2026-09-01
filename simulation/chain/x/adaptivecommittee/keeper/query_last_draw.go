package keeper

import (
	"context"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"chain-five-three/x/adaptivecommittee/types"
)

func (k Keeper) LastDraw(goCtx context.Context, req *types.QueryLastDrawRequest) (*types.QueryLastDrawResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "empty request")
	}
	ctx := sdk.UnwrapSDKContext(goCtx)
	membersCsv, ok := k.GetLastDraw(ctx, req.Tag)
	if !ok {
		membersCsv = ""
	}
	return &types.QueryLastDrawResponse{MembersCsv: membersCsv}, nil
}

package keeper_test

import (
	"testing"

	testkeeper "chain-five-three/testutil/keeper"
	"chain-five-three/x/adaptivecommittee/types"
	"github.com/stretchr/testify/require"
)

func TestGetParams(t *testing.T) {
	k, ctx := testkeeper.AdaptivecommitteeKeeper(t)
	params := types.DefaultParams()

	k.SetParams(ctx, params)

	require.EqualValues(t, params, k.GetParams(ctx))
}

package adaptivecommittee_test

import (
	"testing"

	"chain-five-three/x/adaptivecommittee/types"
	"github.com/stretchr/testify/require"
)

func TestGenesisStateValidate(t *testing.T) {
	genesisState := types.GenesisState{
		Params: types.DefaultParams(),
	}

	require.NoError(t, genesisState.Validate())
	defaultGenesis := types.DefaultGenesis()
	require.NotNil(t, defaultGenesis)
	require.NoError(t, defaultGenesis.Validate())
}

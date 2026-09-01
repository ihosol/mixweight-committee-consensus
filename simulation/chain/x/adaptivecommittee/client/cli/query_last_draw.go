package cli

import (
	"github.com/cosmos/cosmos-sdk/client"
	"github.com/cosmos/cosmos-sdk/client/flags"
	"github.com/spf13/cobra"

	"chain-five-three/x/adaptivecommittee/types"
)

func CmdQueryLastDraw() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "last-draw [tag]",
		Short: "shows the most recently drawn committee members for a given tag",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientCtx, err := client.GetClientQueryContext(cmd)
			if err != nil {
				return err
			}
			queryClient := types.NewQueryClient(clientCtx)
			res, err := queryClient.LastDraw(cmd.Context(), &types.QueryLastDrawRequest{Tag: args[0]})
			if err != nil {
				return err
			}
			return clientCtx.PrintProto(res)
		},
	}

	flags.AddQueryFlagsToCmd(cmd)
	return cmd
}

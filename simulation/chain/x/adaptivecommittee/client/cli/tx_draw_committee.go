package cli

import (
	"strconv"

	"chain-five-three/x/adaptivecommittee/types"
	"github.com/cosmos/cosmos-sdk/client"
	"github.com/cosmos/cosmos-sdk/client/flags"
	"github.com/cosmos/cosmos-sdk/client/tx"
	"github.com/spf13/cast"
	"github.com/spf13/cobra"
)

var _ = strconv.Itoa(0)

func CmdDrawCommittee() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "draw-committee [size] [tag]",
		Short: "Broadcast message draw-committee",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			argSize, err := cast.ToUint64E(args[0])
			if err != nil {
				return err
			}
			argTag := args[1]

			clientCtx, err := client.GetClientTxContext(cmd)
			if err != nil {
				return err
			}

			msg := types.NewMsgDrawCommittee(
				clientCtx.GetFromAddress().String(),
				argSize,
				argTag,
			)
			if err := msg.ValidateBasic(); err != nil {
				return err
			}
			return tx.GenerateOrBroadcastTxCLI(clientCtx, cmd.Flags(), msg)
		},
	}

	flags.AddTxFlagsToCmd(cmd)

	return cmd
}

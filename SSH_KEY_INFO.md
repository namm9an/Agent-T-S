# SSH Key Information for Agent-T-S Project

## Public Key (Add this to GitHub)
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM+eQcmTkZrxd8yPlWFJ7Ed8YVU4q3bAlca4Ha+oW1hs Agent-T-S-project
```

## Key Location
- Private Key: `~/.ssh/agent_ts_key`
- Public Key: `~/.ssh/agent_ts_key.pub`

## How to Add to GitHub
1. Go to: https://github.com/settings/keys
2. Click "New SSH key"
3. Title: "Agent-T-S Project Key"
4. Key type: Authentication Key
5. Paste the public key above
6. Click "Add SSH key"

## Project-Specific Git Remote
After adding the key to GitHub, the project is configured to use:
```bash
git@github-agent-ts:namm9an/Agent-T-S.git
```

This uses a custom SSH config that automatically uses the project-specific key.

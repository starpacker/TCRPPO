# test53 One-Step Delta Amplification

## Goal

Test whether the policy can learn useful local edits when the episode is
reduced to a single required edit:

```text
random natural TCRdb CDR3b -> one substitution -> terminal tFold delta reward
```

This is meant to separate "can the policy find locally beneficial edits?" from
the harder 8-step credit-assignment problem.

## Reward

Reward mode: `tfold_delta_amplified`

Base signal:

```text
delta = final_tfold_logit - initial_tfold_logit
```

Positive deltas are amplified with continuous piecewise slopes:

| Delta interval | Slope |
| --- | ---: |
| 0.0 to 0.5 | 1 |
| 0.5 to 1.0 | 2 |
| 1.0 to 2.0 | 4 |
| > 2.0 | 8 |

Negative deltas remain linear. Tiny deltas below `0.05` receive a small
deadband penalty, which discourages same-residue no-op substitutions.

Naturalness is a hard gate: if the final sequence fails the ESM perplexity
z-score threshold, positive reward is removed and the naturalness penalty is
applied.

## Initial TCRs

The curriculum is fixed to pure L2:

```yaml
curriculum_schedule:
  - {until: null, L0: 0.0, L1: 0.0, L2: 1.0}
```

So initial TCRs are sampled directly from TCRdb, not from mutated L0 binders or
ERGO-selected L1 seeds.

## Targets

The easy run uses three tFold-friendly peptides:

```text
NLVPMVATV
KLGGALQAK
RLRAEAQVK
```

These are the peptides where prior tFold checks showed the clearest positive
reward alignment, so this version should be easier than the 20-target run.

## Launch

```bash
cd /share/liuyutian/tcrppo_v2
scripts/launch_test53_one_step_delta_amp.sh start trace19_one_step_delta_amp <gpu_id>
```

The default budget is 200K timesteps. Because `max_steps=1` scores both the
initial and final TCR every environment step, this is already a dense tFold run.

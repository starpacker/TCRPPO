#!/usr/bin/env python3
"""
Real-time monitoring for trace102 experiment
Tracks key metrics: naturalness pass rate, affinity>0 rate, and both satisfied rate
"""

import re
import time
import sys
from collections import deque

def parse_episode_line(line):
    """Parse episode line to extract metrics"""
    match = re.search(r'Episode\s+(\d+).*R=([-0-9.]+).*Nat=([-0-9.]+).*TargetShort=([-0-9.]+)', line)
    if match:
        ep = int(match.group(1))
        reward = float(match.group(2))
        nat = float(match.group(3))
        target_short = float(match.group(4))
        return {'ep': ep, 'reward': reward, 'nat': nat, 'target_short': target_short}
    return None

def parse_affinity_logit(line):
    """Parse tFold score line to extract affinity logit"""
    match = re.search(r'affinity_logit=([-0-9.]+)', line)
    if match:
        return float(match.group(1))
    return None

def monitor_log(logfile, window=200):
    """Monitor log file and compute statistics"""
    print(f"Monitoring {logfile} (last {window} episodes)")
    print("="*80)

    episodes = deque(maxlen=window)
    affinity_logits = deque(maxlen=500)

    try:
        with open(logfile, 'r') as f:
            # Seek to end and read backwards to get recent data
            f.seek(0, 2)  # Go to end
            file_size = f.tell()
            f.seek(max(0, file_size - 100000), 0)  # Read last ~100KB

            for line in f:
                ep_data = parse_episode_line(line)
                if ep_data:
                    episodes.append(ep_data)

                logit = parse_affinity_logit(line)
                if logit is not None:
                    affinity_logits.append(logit)

        if len(episodes) == 0:
            print("No episode data found yet. Waiting for training to start...")
            return

        # Compute statistics (trace102 uses -0.45 threshold!)
        nat_pass_count = sum(1 for ep in episodes if ep['nat'] > -0.45)
        nat_pass_rate = nat_pass_count / len(episodes) * 100

        target_good_count = sum(1 for ep in episodes if ep['target_short'] < 0.5)
        target_good_rate = target_good_count / len(episodes) * 100

        both_count = sum(1 for ep in episodes if ep['nat'] > -0.45 and ep['target_short'] < 0.5)
        both_rate = both_count / len(episodes) * 100

        avg_reward = sum(ep['reward'] for ep in episodes) / len(episodes)
        avg_nat = sum(ep['nat'] for ep in episodes) / len(episodes)
        avg_target = sum(ep['target_short'] for ep in episodes) / len(episodes)

        # Affinity logit stats
        if len(affinity_logits) > 0:
            aff_above_0 = sum(1 for a in affinity_logits if a > 0.0) / len(affinity_logits) * 100
            aff_above_neg05 = sum(1 for a in affinity_logits if a > -0.5) / len(affinity_logits) * 100
            avg_aff = sum(affinity_logits) / len(affinity_logits)
        else:
            aff_above_0 = aff_above_neg05 = avg_aff = 0.0

        # Print results
        latest_ep = episodes[-1]['ep']
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Latest Episode: {latest_ep}")
        print(f"{'='*80}")
        print(f"📊 Episode Statistics (last {len(episodes)} episodes):")
        print(f"  Reward:           {avg_reward:7.3f}")
        print(f"  Naturalness:      {avg_nat:7.3f}")
        print(f"  TargetShort:      {avg_target:7.3f}")
        print(f"")
        print(f"🎯 Key Metrics (trace102 threshold: -0.45):")
        print(f"  Nat pass rate (>-0.45):       {nat_pass_rate:5.1f}%  ({nat_pass_count}/{len(episodes)})")
        print(f"  Target good rate (<0.5):      {target_good_rate:5.1f}%  ({target_good_count}/{len(episodes)})")
        print(f"  ⭐ Both satisfied:            {both_rate:5.1f}%  ({both_count}/{len(episodes)})")
        print(f"")
        print(f"🧬 Affinity Logit Stats (last {len(affinity_logits)} samples):")
        print(f"  Mean logit:       {avg_aff:7.3f}")
        print(f"  Logit > 0.0:      {aff_above_0:5.1f}%")
        print(f"  Logit > -0.5:     {aff_above_neg05:5.1f}%")
        print(f"")

        # Progress assessment
        print(f"📈 Progress Assessment:")
        if both_rate < 10:
            print(f"  Status: 🔴 EARLY STAGE - Both conditions rarely met")
        elif both_rate < 20:
            print(f"  Status: 🟡 LEARNING - Some progress on both objectives")
        elif both_rate < 35:
            print(f"  Status: 🟢 GOOD - Significant progress!")
        else:
            print(f"  Status: ✅ EXCELLENT - Target achieved!")

        if nat_pass_rate < 40:
            print(f"  → Naturalness needs improvement (target: >55%)")
        if target_good_rate < 40:
            print(f"  → Affinity needs improvement (target: >50%)")

        print(f"{'='*80}\n")

    except FileNotFoundError:
        print(f"ERROR: Log file not found: {logfile}")
        print("Make sure training has started.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == '__main__':
    logfile = 'logs/trace102_nat_gate_from_trace61_train.log'

    if len(sys.argv) > 1:
        logfile = sys.argv[1]

    print(f"trace102 Real-Time Monitor")
    print(f"Log: {logfile}")
    print(f"Naturalness gate threshold: -0.45 (softer than trace100's -0.3)")
    print(f"Refresh every 30 seconds. Press Ctrl+C to stop.")
    print("")

    try:
        while True:
            monitor_log(logfile)
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

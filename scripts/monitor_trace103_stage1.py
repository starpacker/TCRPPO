#!/usr/bin/env python3
"""
Real-time monitoring for trace103 stage1 (pure naturalness pretraining)
Tracks naturalness score improvement over time
"""

import re
import time
import sys
from collections import deque

def parse_episode_line(line):
    """Parse episode line to extract metrics"""
    match = re.search(r'Episode\s+(\d+).*Step\s+(\d+).*R=([-0-9.]+).*Nat=([-0-9.]+)', line)
    if match:
        ep = int(match.group(1))
        step = int(match.group(2))
        reward = float(match.group(3))
        nat = float(match.group(4))
        return {'ep': ep, 'step': step, 'reward': reward, 'nat': nat}
    return None

def monitor_log(logfile, window=200):
    """Monitor log file and compute statistics"""
    print(f"Monitoring {logfile} (last {window} episodes)")
    print("="*80)

    episodes = deque(maxlen=window)

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

        if len(episodes) == 0:
            print("No episode data found yet. Waiting for training to start...")
            return

        # Compute statistics
        avg_reward = sum(ep['reward'] for ep in episodes) / len(episodes)
        avg_nat = sum(ep['nat'] for ep in episodes) / len(episodes)
        max_nat = max(ep['nat'] for ep in episodes)
        min_nat = min(ep['nat'] for ep in episodes)

        # Count high-quality episodes (naturalness > 0.8)
        high_nat_count = sum(1 for ep in episodes if ep['nat'] > 0.8)
        high_nat_rate = high_nat_count / len(episodes) * 100

        # Count medium-quality episodes (naturalness > 0.5)
        med_nat_count = sum(1 for ep in episodes if ep['nat'] > 0.5)
        med_nat_rate = med_nat_count / len(episodes) * 100

        # Print results
        latest_ep = episodes[-1]['ep']
        latest_step = episodes[-1]['step']
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Latest Episode: {latest_ep}, Step: {latest_step}")
        print(f"{'='*80}")
        print(f"📊 Episode Statistics (last {len(episodes)} episodes):")
        print(f"  Reward (=Nat):    {avg_reward:7.3f}")
        print(f"  Naturalness:      {avg_nat:7.3f}  (range: {min_nat:.3f} ~ {max_nat:.3f})")
        print(f"")
        print(f"🎯 Quality Metrics:")
        print(f"  Nat > 0.8 (high):     {high_nat_rate:5.1f}%  ({high_nat_count}/{len(episodes)})")
        print(f"  Nat > 0.5 (medium):   {med_nat_rate:5.1f}%  ({med_nat_count}/{len(episodes)})")
        print(f"")

        # Progress assessment
        print(f"📈 Progress Assessment:")
        if avg_nat < 0.3:
            print(f"  Status: 🔴 EARLY STAGE - Low naturalness scores")
        elif avg_nat < 0.6:
            print(f"  Status: 🟡 LEARNING - Improving naturalness")
        elif avg_nat < 0.8:
            print(f"  Status: 🟢 GOOD - Strong naturalness")
        else:
            print(f"  Status: ✅ EXCELLENT - High-quality natural sequences!")

        # Target for stage1: mean naturalness > 0.8 (like trace97)
        if avg_nat < 0.7:
            print(f"  → Target: reach mean naturalness > 0.8 before stage2")
        if latest_step < 10000:
            print(f"  → Continue training to 10,000 steps (currently {latest_step})")
        else:
            print(f"  → ✅ Stage1 complete! Ready for stage2 delta reward finetune")

        print(f"{'='*80}\n")

    except FileNotFoundError:
        print(f"ERROR: Log file not found: {logfile}")
        print("Make sure stage1 training has started.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == '__main__':
    logfile = 'logs/trace103_stage1_nat_pretrain_train.log'

    if len(sys.argv) > 1:
        logfile = sys.argv[1]

    print(f"trace103 Stage 1 Real-Time Monitor")
    print(f"Log: {logfile}")
    print(f"Mode: Pure naturalness pretraining (10k steps)")
    print(f"Refresh every 30 seconds. Press Ctrl+C to stop.")
    print("")

    try:
        while True:
            monitor_log(logfile)
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

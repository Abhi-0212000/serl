#!/usr/bin/env python3
"""
Minimal teleoperation example - single leader-follower pair
Use this for quick testing or as a starting template
"""

import time
import numpy as np
import trossen_arm

# Configuration
LEADER_IP = '192.168.1.2'
FOLLOWER_IP = '192.168.1.3'
DURATION = 20  # seconds
FORCE_GAIN = 0.1

print("="*70)
print("MINIMAL TELEOPERATION EXAMPLE")
print("="*70)
print(f"\nLeader: {LEADER_IP}")
print(f"Follower: {FOLLOWER_IP}")
print(f"Duration: {DURATION}s")
print(f"Force gain: {FORCE_GAIN}")
print()

try:
    # Initialize
    print("Initializing drivers...")
    leader = trossen_arm.TrossenArmDriver()
    follower = trossen_arm.TrossenArmDriver()
    
    # Configure
    print("Configuring robots...")
    leader.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        LEADER_IP,
        False
    )
    follower.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_follower,
        FOLLOWER_IP,
        False
    )
    
    # Home position
    print("Moving to home...")
    home = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0])
    
    leader.set_all_modes(trossen_arm.Mode.position)
    leader.set_all_positions(home, 2.0, True)
    
    follower.set_all_modes(trossen_arm.Mode.position)
    follower.set_all_positions(home, 2.0, True)
    
    # Start teleoperation
    print("\nStarting teleoperation...")
    input("Press ENTER when ready...")
    
    leader.set_all_modes(trossen_arm.Mode.external_effort)
    follower.set_all_modes(trossen_arm.Mode.position)
    
    start = time.time()
    end = start + DURATION
    
    print(f"Teleoperation active for {DURATION}s...")
    
    while time.time() < end:
        # Force feedback: follower -> leader
        leader.set_all_external_efforts(
            -FORCE_GAIN * np.array(follower.get_all_external_efforts()),
            0.0,
            False
        )
        
        # Position tracking: leader -> follower
        follower.set_all_positions(
            leader.get_all_positions(),
            0.0,
            False,
            leader.get_all_velocities()
        )
    
    print("Teleoperation complete!")
    
    # Return home
    print("Returning home...")
    leader.set_all_modes(trossen_arm.Mode.position)
    leader.set_all_positions(home, 2.0, True)
    
    follower.set_all_modes(trossen_arm.Mode.position)
    follower.set_all_positions(home, 2.0, True)
    
    # Sleep
    print("Moving to sleep position...")
    sleep = np.zeros(7)
    leader.set_all_positions(sleep, 2.0, True)
    follower.set_all_positions(sleep, 2.0, True)
    
    print("\n✓ Success!")
    
except KeyboardInterrupt:
    print("\n\nInterrupted by user!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\nCleaning up...")
    try:
        leader.cleanup()
        follower.cleanup()
    except:
        pass

print("Done!")

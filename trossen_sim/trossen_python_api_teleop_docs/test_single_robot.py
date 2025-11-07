#!/usr/bin/env python3
"""
Simple test script to verify individual robot connections.
Use this to test each robot before running the full teleoperation.
"""

import trossen_arm
import numpy as np
import argparse


def test_robot(ip_address: str, robot_type: str, model: str = 'wxai_v0'):
    """
    Test a single robot connection and basic movements.
    
    Args:
        ip_address: IP address of the robot
        robot_type: 'leader' or 'follower'
        model: Robot model
    """
    print(f"\n{'='*70}")
    print(f"Testing {robot_type.upper()} robot at {ip_address}")
    print(f"{'='*70}\n")
    
    # Set model
    if model == 'wxai_v0':
        robot_model = trossen_arm.Model.wxai_v0
    elif model == 'vxai_v0_right':
        robot_model = trossen_arm.Model.vxai_v0_right
    elif model == 'vxai_v0_left':
        robot_model = trossen_arm.Model.vxai_v0_left
    else:
        print(f"Unknown model: {model}")
        return False
    
    # Set end effector
    if robot_type == 'leader':
        end_effector = trossen_arm.StandardEndEffector.wxai_v0_leader
    elif robot_type == 'follower':
        end_effector = trossen_arm.StandardEndEffector.wxai_v0_follower
    else:
        print(f"Unknown robot type: {robot_type}")
        return False
    
    try:
        # Initialize driver
        print("1. Initializing driver...")
        driver = trossen_arm.TrossenArmDriver()
        
        # Configure
        print(f"2. Configuring driver for {ip_address}...")
        driver.configure(
            robot_model,
            end_effector,
            ip_address,
            False,
            20.0
        )
        
        print("   ✓ Configuration successful!")
        
        # Get basic info
        print("\n3. Robot Information:")
        print(f"   - Number of joints: {driver.get_num_joints()}")
        print(f"   - Driver version: {driver.get_driver_version()}")
        print(f"   - Controller version: {driver.get_controller_version()}")
        print(f"   - Is configured: {driver.get_is_configured()}")
        
        # Get current state
        print("\n4. Current State:")
        positions = driver.get_all_positions()
        velocities = driver.get_all_velocities()
        efforts = driver.get_all_efforts()
        external_efforts = driver.get_all_external_efforts()
        
        print(f"   - Positions: {[f'{p:.3f}' for p in positions]}")
        print(f"   - Velocities: {[f'{v:.3f}' for v in velocities]}")
        print(f"   - Efforts: {[f'{e:.3f}' for e in efforts]}")
        print(f"   - External efforts: {[f'{e:.3f}' for e in external_efforts]}")
        
        # Check temperatures
        rotor_temps = driver.get_all_rotor_temperatures()
        driver_temps = driver.get_all_driver_temperatures()
        print(f"   - Rotor temperatures: {[f'{t:.1f}°C' for t in rotor_temps]}")
        print(f"   - Driver temperatures: {[f'{t:.1f}°C' for t in driver_temps]}")
        
        # Test basic movement
        print("\n5. Testing basic movement...")
        input("   Press ENTER to move to home position...")
        
        # Move to home
        driver.set_all_modes(trossen_arm.Mode.position)
        home_positions = np.zeros(driver.get_num_joints())
        home_positions[1] = np.pi / 2
        home_positions[2] = np.pi / 2
        
        print("   Moving to home position...")
        driver.set_all_positions(home_positions, 2.0, True)
        print("   ✓ Reached home position!")
        
        # Get new positions
        new_positions = driver.get_all_positions()
        print(f"   - New positions: {[f'{p:.3f}' for p in new_positions]}")
        
        # Move back to sleep
        print("\n6. Moving back to sleep position...")
        sleep_positions = np.zeros(driver.get_num_joints())
        driver.set_all_positions(sleep_positions, 2.0, True)
        print("   ✓ Reached sleep position!")
        
        # Test gripper (if leader)
        if robot_type == 'leader':
            print("\n7. Testing gripper force control...")
            input("   Press ENTER to open gripper with force control...")
            
            driver.set_gripper_mode(trossen_arm.Mode.external_effort)
            driver.set_gripper_external_effort(20.0, 2.0, True)
            print("   ✓ Gripper opened!")
            
            input("   Press ENTER to close gripper with force control...")
            driver.set_gripper_external_effort(-20.0, 2.0, True)
            print("   ✓ Gripper closed!")
        
        # Cleanup
        print("\n8. Cleaning up...")
        driver.cleanup()
        print("   ✓ Cleanup complete!")
        
        print(f"\n{'='*70}")
        print(f"✓ ALL TESTS PASSED for {robot_type} at {ip_address}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n{'='*70}")
        print(f"✗ TESTS FAILED for {robot_type} at {ip_address}")
        print(f"{'='*70}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test individual robot connections'
    )
    parser.add_argument(
        '--ip',
        required=True,
        help='IP address of the robot to test'
    )
    parser.add_argument(
        '--type',
        required=True,
        choices=['leader', 'follower'],
        help='Type of robot (leader or follower)'
    )
    parser.add_argument(
        '--model',
        default='wxai_v0',
        choices=['wxai_v0', 'vxai_v0_right', 'vxai_v0_left'],
        help='Robot model'
    )
    
    args = parser.parse_args()
    
    success = test_robot(args.ip, args.type, args.model)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

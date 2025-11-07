#!/usr/bin/env python3
"""
Single Robot Teleoperation Script
1 Leader + 1 Follower

This script implements bilateral teleoperation with force feedback for one pair
of leader-follower robot arms.
python3 single_robot_teleop.py --leader-ip 192.168.1.4 --follower-ip 192.168.1.5

"""

import time
import numpy as np
import trossen_arm
import argparse
import logging
from typing import List


class SingleRobotTeleop:
    """Manages teleoperation for 1 leader + 1 follower robot setup"""
    
    def __init__(self, 
                 leader_ip: str = '192.168.1.4',
                 follower_ip: str = '192.168.1.5',
                 force_feedback_gain: float = 0.1,
                 model: str = 'wxai_v0'):
        """
        Initialize single robot teleoperation system.
        
        Args:
            leader_ip: IP address of leader robot
            follower_ip: IP address of follower robot
            force_feedback_gain: Gain for force feedback (0.05-0.2 typical)
            model: Robot model ('wxai_v0', 'vxai_v0_right', 'vxai_v0_left')
        """
        self.leader_ip = leader_ip
        self.follower_ip = follower_ip
        self.force_feedback_gain = force_feedback_gain
        
        # Set model
        if model == 'wxai_v0':
            self.model = trossen_arm.Model.wxai_v0
        elif model == 'vxai_v0_right':
            self.model = trossen_arm.Model.vxai_v0_right
        elif model == 'vxai_v0_left':
            self.model = trossen_arm.Model.vxai_v0_left
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Initialize drivers
        self.driver_leader = None
        self.driver_follower = None
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the teleoperation system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SingleRobotTeleop')
        
    def initialize_drivers(self):
        """Initialize and configure both robot drivers"""
        self.logger.info("Initializing drivers...")
        
        try:
            # Create driver instances
            self.driver_leader = trossen_arm.TrossenArmDriver()
            self.driver_follower = trossen_arm.TrossenArmDriver()
            
            # Configure leader
            self.logger.info(f"Configuring Leader at {self.leader_ip}...")
            self.driver_leader.configure(
                self.model,
                trossen_arm.StandardEndEffector.wxai_v0_leader,
                self.leader_ip,
                False
            )
            
            # Configure follower
            self.logger.info(f"Configuring Follower at {self.follower_ip}...")
            self.driver_follower.configure(
                self.model,
                trossen_arm.StandardEndEffector.wxai_v0_follower,
                self.follower_ip,
                False
            )
            
            self.logger.info("Both drivers configured successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize drivers: {e}")
            return False
    
    def move_to_home(self):
        """Move both robots to home position"""
        self.logger.info("Moving both robots to home position...")
        
        # Get number of joints
        num_joints = self.driver_leader.get_num_joints()
        
        # Define home position
        home_positions = np.zeros(num_joints)
        home_positions[1] = np.pi / 2
        home_positions[2] = np.pi / 2
        
        # Set both robots to position mode
        for driver in self._get_all_drivers():
            driver.set_all_modes(trossen_arm.Mode.position)
        
        # Move both robots to home
        for name, driver in [("Leader", self.driver_leader), ("Follower", self.driver_follower)]:
            self.logger.info(f"Moving {name} to home...")
            driver.set_all_positions(home_positions, 2.0, True)
        
        self.logger.info("Both robots at home position")
    
    def move_to_sleep(self):
        """Move both robots to sleep position (all zeros)"""
        self.logger.info("Moving both robots to sleep position...")
        
        num_joints = self.driver_leader.get_num_joints()
        sleep_positions = np.zeros(num_joints)
        
        # Set both robots to position mode
        for driver in self._get_all_drivers():
            driver.set_all_modes(trossen_arm.Mode.position)
        
        # Move both robots to sleep
        for name, driver in [("Leader", self.driver_leader), ("Follower", self.driver_follower)]:
            self.logger.info(f"Moving {name} to sleep...")
            driver.set_all_positions(sleep_positions, 2.0, True)
        
        self.logger.info("Both robots in sleep position")
    
    def start_teleoperation(self, duration: float = 60.0):
        """
        Start bilateral teleoperation with force feedback.
        
        Args:
            duration: Duration of teleoperation in seconds
        """
        self.logger.info(f"Starting teleoperation for {duration} seconds...")
        self.logger.info("Leader will provide force feedback, follower will track leader")
        self.logger.info("IMPORTANT: Release the leader when time expires!")
        
        # Wait a moment for user to prepare
        time.sleep(1)
        
        # Set leader to external effort mode (for force feedback)
        self.logger.info("Setting leader to external effort mode...")
        self.driver_leader.set_all_modes(trossen_arm.Mode.external_effort)
        
        # Set follower to position mode (to track leader)
        self.logger.info("Setting follower to position mode...")
        self.driver_follower.set_all_modes(trossen_arm.Mode.position)
        
        # Teleoperation loop
        start_time = time.time()
        end_time = start_time + duration
        loop_count = 0
        
        self.logger.info("Teleoperation active!")
        
        try:
            while time.time() < end_time:
                # Update leader-follower pair
                self._update_leader_follower_pair(
                    self.driver_leader,
                    self.driver_follower
                )
                
                loop_count += 1
                
                # Log status every 5 seconds
                if loop_count % 500 == 0:
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    self.logger.info(f"Time remaining: {remaining:.1f}s")
                    
        except KeyboardInterrupt:
            self.logger.warning("Teleoperation interrupted by user!")
        except Exception as e:
            self.logger.error(f"Error during teleoperation: {e}")
            raise
        
        self.logger.info(f"Teleoperation complete! Executed {loop_count} control loops")
    
    def _update_leader_follower_pair(self, 
                                     leader_driver: trossen_arm.TrossenArmDriver,
                                     follower_driver: trossen_arm.TrossenArmDriver):
        """
        Update the leader-follower pair.
        
        Args:
            leader_driver: Leader robot driver
            follower_driver: Follower robot driver
        """
        # Get external efforts from follower
        follower_efforts = np.array(follower_driver.get_all_external_efforts())
        
        # Apply force feedback to leader (negative gain for proper feel)
        leader_driver.set_all_external_efforts(
            -self.force_feedback_gain * follower_efforts,
            0.0,  # Immediate
            False  # Non-blocking
        )
        
        # Get leader state
        leader_positions = leader_driver.get_all_positions()
        leader_velocities = leader_driver.get_all_velocities()
        
        # Command follower to track leader
        follower_driver.set_all_positions(
            leader_positions,
            0.0,  # Immediate
            False,  # Non-blocking
            leader_velocities  # Feedforward velocities for smooth tracking
        )
    
    def _get_all_drivers(self) -> List[trossen_arm.TrossenArmDriver]:
        """Get list of all drivers"""
        return [
            self.driver_leader,
            self.driver_follower
        ]
    
    def cleanup(self):
        """Cleanup all drivers"""
        self.logger.info("Cleaning up drivers...")
        
        for driver in self._get_all_drivers():
            if driver is not None:
                try:
                    driver.cleanup()
                except Exception as e:
                    self.logger.error(f"Error during cleanup: {e}")
        
        self.logger.info("Cleanup complete")
    
    def print_status(self):
        """Print status of both robots"""
        print("\n" + "="*70)
        print("ROBOT STATUS")
        print("="*70)
        
        for name, driver in [("Leader", self.driver_leader), ("Follower", self.driver_follower)]:
            print(f"\n{name}:")
            print(f"  Configured: {driver.get_is_configured()}")
            print(f"  Num Joints: {driver.get_num_joints()}")
            print(f"  Driver Version: {driver.get_driver_version()}")
            print(f"  Controller Version: {driver.get_controller_version()}")
            
            positions = driver.get_all_positions()
            print(f"  Positions: {[f'{p:.3f}' for p in positions]}")
            
            temps = driver.get_all_rotor_temperatures()
            print(f"  Temperatures: {[f'{t:.1f}°C' for t in temps]}")
        
        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Single Robot Teleoperation'
    )
    parser.add_argument(
        '--leader-ip', 
        default='192.168.1.4',
        help='IP address of leader robot'
    )
    parser.add_argument(
        '--follower-ip', 
        default='192.168.1.5',
        help='IP address of follower robot'
    )
    parser.add_argument(
        '--duration', 
        type=float, 
        default=60.0,
        help='Duration of teleoperation in seconds'
    )
    parser.add_argument(
        '--force-gain', 
        type=float, 
        default=0.1,
        help='Force feedback gain (0.05-0.2 typical)'
    )
    parser.add_argument(
        '--model', 
        default='wxai_v0',
        choices=['wxai_v0', 'vxai_v0_right', 'vxai_v0_left'],
        help='Robot model'
    )
    parser.add_argument(
        '--no-home',
        action='store_true',
        help='Skip moving to home position'
    )
    parser.add_argument(
        '--no-sleep',
        action='store_true',
        help='Skip moving to sleep position at end'
    )
    
    args = parser.parse_args()
    
    # Create teleoperation system
    teleop = SingleRobotTeleop(
        leader_ip=args.leader_ip,
        follower_ip=args.follower_ip,
        force_feedback_gain=args.force_gain,
        model=args.model
    )
    
    try:
        # Initialize both robots
        if not teleop.initialize_drivers():
            print("Failed to initialize drivers. Exiting.")
            return 1
        
        # Print status
        teleop.print_status()
        
        # Move to home position
        if not args.no_home:
            teleop.move_to_home()
        
        # Start teleoperation
        print("\n" + "="*70)
        print("TELEOPERATION STARTING")
        print("="*70)
        print(f"Duration: {args.duration} seconds")
        print(f"Force feedback gain: {args.force_gain}")
        print("\nLeader will provide force feedback.")
        print("Follower will track leader positions.")
        print("\nPress Ctrl+C to stop early.")
        print("="*70 + "\n")
        
        input("Press ENTER to start teleoperation...")
        
        teleop.start_teleoperation(duration=args.duration)
        
        # Return to home
        if not args.no_home:
            print("\nReturning to home position...")
            teleop.move_to_home()
        
        # Move to sleep position
        if not args.no_sleep:
            print("\nMoving to sleep position...")
            teleop.move_to_sleep()
        
        print("\n" + "="*70)
        print("TELEOPERATION COMPLETE")
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        return 1
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Always cleanup
        teleop.cleanup()


if __name__ == '__main__':
    exit(main())
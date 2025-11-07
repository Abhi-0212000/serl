#!/usr/bin/env python3
"""
Dual Robot Teleoperation Script for WINDOWX Setup
2 Leaders + 2 Followers

This script implements bilateral teleoperation with force feedback for two pairs
of leader-follower robot arms.
"""

import time
import numpy as np
import trossen_arm
import argparse
import logging
from typing import List


class DualRobotTeleop:
    """Manages teleoperation for 2 leader + 2 follower robot setup"""
    
    def __init__(self, 
                 leader_1_ip: str = '192.168.1.2',
                 leader_2_ip: str = '192.168.1.3',
                 follower_1_ip: str = '192.168.1.4',
                 follower_2_ip: str = '192.168.1.5',
                 force_feedback_gain: float = 0.1,
                 model: str = 'wxai_v0'):
        """
        Initialize dual robot teleoperation system.
        
        Args:
            leader_1_ip: IP address of first leader robot
            leader_2_ip: IP address of second leader robot
            follower_1_ip: IP address of first follower robot
            follower_2_ip: IP address of second follower robot
            force_feedback_gain: Gain for force feedback (0.05-0.2 typical)
            model: Robot model ('wxai_v0', 'vxai_v0_right', 'vxai_v0_left')
        """
        self.leader_1_ip = leader_1_ip
        self.leader_2_ip = leader_2_ip
        self.follower_1_ip = follower_1_ip
        self.follower_2_ip = follower_2_ip
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
        self.driver_leader_1 = None
        self.driver_leader_2 = None
        self.driver_follower_1 = None
        self.driver_follower_2 = None
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the teleoperation system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DualRobotTeleop')
        
    def initialize_drivers(self):
        """Initialize and configure all four robot drivers"""
        self.logger.info("Initializing drivers...")
        
        try:
            # Create driver instances
            self.driver_leader_1 = trossen_arm.TrossenArmDriver()
            self.driver_leader_2 = trossen_arm.TrossenArmDriver()
            self.driver_follower_1 = trossen_arm.TrossenArmDriver()
            self.driver_follower_2 = trossen_arm.TrossenArmDriver()
            
            # Configure leaders
            self.logger.info(f"Configuring Leader 1 at {self.leader_1_ip}...")
            self.driver_leader_1.configure(
                self.model,
                trossen_arm.StandardEndEffector.wxai_v0_leader,
                self.leader_1_ip,
                False
            )
            
            self.logger.info(f"Configuring Leader 2 at {self.leader_2_ip}...")
            self.driver_leader_2.configure(
                self.model,
                trossen_arm.StandardEndEffector.wxai_v0_leader,
                self.leader_2_ip,
                False
            )
            
            # Configure followers
            self.logger.info(f"Configuring Follower 1 at {self.follower_1_ip}...")
            self.driver_follower_1.configure(
                self.model,
                trossen_arm.StandardEndEffector.wxai_v0_follower,
                self.follower_1_ip,
                False
            )
            
            self.logger.info(f"Configuring Follower 2 at {self.follower_2_ip}...")
            self.driver_follower_2.configure(
                self.model,
                trossen_arm.StandardEndEffector.wxai_v0_follower,
                self.follower_2_ip,
                False
            )
            
            self.logger.info("All drivers configured successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize drivers: {e}")
            return False
    
    def move_to_home(self):
        """Move all robots to home position"""
        self.logger.info("Moving all robots to home position...")
        
        # Get number of joints
        num_joints = self.driver_leader_1.get_num_joints()
        
        # Define home position
        home_positions = np.zeros(num_joints)
        home_positions[1] = np.pi / 2
        home_positions[2] = np.pi / 2
        
        # Set all robots to position mode
        for driver in self._get_all_drivers():
            driver.set_all_modes(trossen_arm.Mode.position)
        
        # Move all robots to home
        for i, driver in enumerate(self._get_all_drivers(), 1):
            self.logger.info(f"Moving robot {i} to home...")
            driver.set_all_positions(home_positions, 2.0, True)
        
        self.logger.info("All robots at home position")
    
    def move_to_sleep(self):
        """Move all robots to sleep position (all zeros)"""
        self.logger.info("Moving all robots to sleep position...")
        
        num_joints = self.driver_leader_1.get_num_joints()
        sleep_positions = np.zeros(num_joints)
        
        # Set all robots to position mode
        for driver in self._get_all_drivers():
            driver.set_all_modes(trossen_arm.Mode.position)
        
        # Move all robots to sleep
        for i, driver in enumerate(self._get_all_drivers(), 1):
            self.logger.info(f"Moving robot {i} to sleep...")
            driver.set_all_positions(sleep_positions, 2.0, True)
        
        self.logger.info("All robots in sleep position")
    
    def start_teleoperation(self, duration: float = 60.0):
        """
        Start bilateral teleoperation with force feedback.
        
        Args:
            duration: Duration of teleoperation in seconds
        """
        self.logger.info(f"Starting teleoperation for {duration} seconds...")
        self.logger.info("Leaders will provide force feedback, followers will track leaders")
        self.logger.info("IMPORTANT: Release the leaders when time expires!")
        
        # Wait a moment for user to prepare
        time.sleep(1)
        
        # Set leaders to external effort mode (for force feedback)
        self.logger.info("Setting leaders to external effort mode...")
        self.driver_leader_1.set_all_modes(trossen_arm.Mode.external_effort)
        self.driver_leader_2.set_all_modes(trossen_arm.Mode.external_effort)
        
        # Set followers to position mode (to track leaders)
        self.logger.info("Setting followers to position mode...")
        self.driver_follower_1.set_all_modes(trossen_arm.Mode.position)
        self.driver_follower_2.set_all_modes(trossen_arm.Mode.position)
        
        # Teleoperation loop
        start_time = time.time()
        end_time = start_time + duration
        loop_count = 0
        
        self.logger.info("Teleoperation active!")
        
        try:
            while time.time() < end_time:
                # Leader 1 <-> Follower 1
                self._update_leader_follower_pair(
                    self.driver_leader_1,
                    self.driver_follower_1
                )
                
                # Leader 2 <-> Follower 2
                self._update_leader_follower_pair(
                    self.driver_leader_2,
                    self.driver_follower_2
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
        Update one leader-follower pair.
        
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
            self.driver_leader_1,
            self.driver_leader_2,
            self.driver_follower_1,
            self.driver_follower_2
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
        """Print status of all robots"""
        print("\n" + "="*70)
        print("ROBOT STATUS")
        print("="*70)
        
        for i, (name, driver) in enumerate([
            ("Leader 1", self.driver_leader_1),
            ("Leader 2", self.driver_leader_2),
            ("Follower 1", self.driver_follower_1),
            ("Follower 2", self.driver_follower_2)
        ], 1):
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
        description='Dual Robot Teleoperation for WINDOWX Setup'
    )
    parser.add_argument(
        '--leader1-ip', 
        default='192.168.1.2',
        help='IP address of leader 1 robot'
    )
    parser.add_argument(
        '--leader2-ip', 
        default='192.168.1.3',
        help='IP address of leader 2 robot'
    )
    parser.add_argument(
        '--follower1-ip', 
        default='192.168.1.4',
        help='IP address of follower 1 robot'
    )
    parser.add_argument(
        '--follower2-ip', 
        default='192.168.1.5',
        help='IP address of follower 2 robot'
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
    teleop = DualRobotTeleop(
        leader_1_ip=args.leader1_ip,
        leader_2_ip=args.leader2_ip,
        follower_1_ip=args.follower1_ip,
        follower_2_ip=args.follower2_ip,
        force_feedback_gain=args.force_gain,
        model=args.model
    )
    
    try:
        # Initialize all robots
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
        print("\nLeaders will provide force feedback.")
        print("Followers will track leader positions.")
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

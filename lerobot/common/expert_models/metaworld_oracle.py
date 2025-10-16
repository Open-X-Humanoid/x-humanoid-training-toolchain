"""MetaWorld built-in oracle expert model."""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from .base import ExpertModel
import logging

logger = logging.getLogger(__name__)

class MetaWorldOracleExpert(ExpertModel):
    """Expert model using MetaWorld's built-in oracle policies.
    
    These are the official scripted policies provided by MetaWorld that
    achieve near-perfect performance on their respective tasks.
    """
    
    def __init__(self, task_name: str, **kwargs):
        """Initialize MetaWorld oracle expert.
        
        Args:
            task_name: Name of the MetaWorld task (e.g., 'reach-v3')
        """
        super().__init__(task_name, **kwargs)
        self.task_name = task_name
        self.policy = None
        self._policy_map = None
        
    def _get_policy_map(self):
        """Lazy load policy map to avoid import errors."""
        if self._policy_map is None:
            from metaworld.policies import (
                SawyerReachV3Policy,
                SawyerPushV3Policy, 
                SawyerPickPlaceV3Policy,
                SawyerDoorOpenV3Policy,
                SawyerDrawerOpenV3Policy,
                SawyerDrawerCloseV3Policy,
                SawyerButtonPressTopdownV3Policy,
                SawyerButtonPressV3Policy,
                SawyerWindowOpenV3Policy,
                SawyerWindowCloseV3Policy,
                SawyerPegInsertionSideV3Policy,
            )
            
            self._policy_map = {
                'reach-v3': SawyerReachV3Policy,
                'push-v3': SawyerPushV3Policy,
                'pick-place-v3': SawyerPickPlaceV3Policy,
                'door-open-v3': SawyerDoorOpenV3Policy,
                'drawer-open-v3': SawyerDrawerOpenV3Policy,
                'drawer-close-v3': SawyerDrawerCloseV3Policy,
                'button-press-topdown-v3': SawyerButtonPressTopdownV3Policy,
                'button-press-v3': SawyerButtonPressV3Policy,
                'window-open-v3': SawyerWindowOpenV3Policy,
                'window-close-v3': SawyerWindowCloseV3Policy,
                'peg-insert-side-v3': SawyerPegInsertionSideV3Policy,
            }
        return self._policy_map
        
    def initialize(self, env: Any) -> None:
        """Initialize the oracle policy.
        
        Args:
            env: MetaWorld environment instance
        """
        policy_map = self._get_policy_map()
        
        if self.task_name not in policy_map:
            available_tasks = list(policy_map.keys())
            raise ValueError(
                f"No oracle policy available for task '{self.task_name}'. "
                f"Available tasks: {available_tasks}"
            )
        
        policy_class = policy_map[self.task_name]
        self.policy = policy_class()
        
        logger.info(f"Initialized MetaWorld oracle expert for {self.task_name}")
        logger.info(f"Using policy: {policy_class.__name__}")
        
    def generate_action(self, observation: np.ndarray) -> np.ndarray:
        """Generate expert action using MetaWorld's oracle policy.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Expert action
        """
        if self.policy is None:
            raise RuntimeError("Policy not initialized. Call initialize() first.")
        
        # MetaWorld policies expect 39-dimensional observations
        if isinstance(observation, dict):
            # Handle dict observations from wrapped environments
            if 'state' in observation:
                obs = observation['state']
            else:
                obs = observation.get('observation', observation)
        else:
            obs = observation
            
        # Ensure observation is the right shape
        if obs.shape[0] != 39:
            raise ValueError(f"Expected 39-dim observation, got {obs.shape}")
        
        # Get action from oracle policy
        action = self.policy.get_action(obs)
        
        # MetaWorld policies return 4D actions, but environments expect 4D
        if action.shape[0] > 4:
            action = action[:4]
            
        return action
    
    def generate_trajectory(
        self, 
        env: Any, 
        max_steps: int = 200,
        render: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """Generate a complete trajectory using the oracle policy.
        
        Args:
            env: MetaWorld environment
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            
        Returns:
            Tuple of (success, trajectory_dict)
        """
        if self.policy is None:
            self.initialize(env)
        
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        
        obs, _ = env.reset()
        success = False
        
        for step in range(max_steps):
            # Store observation
            observations.append(obs.copy() if isinstance(obs, np.ndarray) else obs)
            
            # Generate action
            action = self.generate_action(obs)
            actions.append(action.copy())
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            next_observations.append(next_obs.copy() if isinstance(next_obs, np.ndarray) else next_obs)
            done = terminated or truncated
            dones.append(done)
            
            # Check success
            if info.get('success', 0.0) > 0:
                success = True
            
            # Render if requested
            if render:
                env.render()
            
            obs = next_obs
            
            if done:
                break
        
        trajectory = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_observations': np.array(next_observations),
            'dones': np.array(dones),
            'success': success,
            'length': len(observations)
        }
        
        return success, trajectory
    
    def reset(self) -> None:
        """Reset the expert (no internal state for oracle policies)."""
        pass  # Oracle policies are stateless

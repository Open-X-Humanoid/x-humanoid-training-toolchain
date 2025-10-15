"""Base class for expert models that generate trajectories."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class ExpertModel(ABC):
    """Abstract base class for expert trajectory generators."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize the expert model.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        self.initialized = False
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the model with any necessary resources."""
        pass
    
    @abstractmethod
    def generate_action(
        self, 
        observation: np.ndarray,
        task_description: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate an expert action given an observation.
        
        Args:
            observation: Current environment observation
            task_description: Optional natural language task description
            **kwargs: Additional model-specific arguments
            
        Returns:
            Action to take in the environment
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state of the expert."""
        pass
    
    def generate_trajectory(
        self,
        env,
        max_steps: int = 500,
        task_description: Optional[str] = None,
        render: bool = False
    ) -> Tuple[bool, Dict[str, np.ndarray]]:
        """Generate a complete trajectory in an environment.
        
        Args:
            env: Gymnasium environment
            max_steps: Maximum steps in the trajectory
            task_description: Optional task description
            render: Whether to render the environment
            
        Returns:
            Tuple of (success, trajectory_dict) where trajectory_dict contains
            observations, actions, rewards, etc.
        """
        observations = []
        actions = []
        rewards = []
        dones = []
        infos = []
        
        obs, info = env.reset()
        self.reset()
        
        for step in range(max_steps):
            # Generate expert action
            action = self.generate_action(obs, task_description)
            
            # Store trajectory data
            observations.append(obs)
            actions.append(action)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)
            
            if render:
                env.render()
            
            if terminated or truncated:
                break
            
            obs = next_obs
        
        # Check if task was successful
        success = info.get("success", False) if info else False
        
        trajectory = {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "success": success,
            "infos": infos
        }
        
        return success, trajectory
